# ── Circular-padded convolution ────────────────────────────────────────────────
# Non-mutating cat-based implementation; avoids NNlib.pad_circular which lacks a
# Reactant/XLA override and is not differentiable through Enzyme.

function _circ_pad_2d(x::AbstractArray, p::Int)
    x = cat(x[end-p+1:end, :, :, :], x, x[1:p, :, :, :]; dims=1)
    x = cat(x[:, end-p+1:end, :, :], x, x[:, 1:p, :, :]; dims=2)
    return x
end

struct CircConv{C} <: Lux.AbstractLuxContainerLayer{(:conv,)}
    conv::C
    pad::Int
end

function CircConv(in_ch::Int, out_ch::Int, k::Int; bias::Bool=true)
    CircConv(Conv((k, k), in_ch => out_ch; pad=0, use_bias=bias), k ÷ 2)
end

function (l::CircConv)(x, ps, st)
    y, new_conv_st = l.conv(_circ_pad_2d(x, l.pad), ps.conv, st.conv)
    return y, (conv=new_conv_st,)
end

# ── Spectral 2× upsampling ─────────────────────────────────────────────────────
# Zero-pads in frequency space then iFFTs. The irfft convention divides by the
# output size, so the correctly-scaled interpolant needs a factor of 4 (= 2² for
# doubling both spatial dimensions).
#
# Zero blocks are derived from x_hat via `.* false` rather than `zeros(T, ...)`
# so they stay on-device (required for Reactant/XLA gradient tracing).

function spectral_upsample_2x(x::AbstractArray{T,4}) where T
    H, W, C, B = size(x)
    x_flat = reshape(x, H, W, C * B)
    x_hat  = rfft(x_flat, 1:2)   # (H÷2+1, W, C*B) complex

    hx = W ÷ 2   # half x-frequencies to keep (left and right of DC)
    hy = H ÷ 2   # non-Nyquist y-frequency rows to keep

    # top: positive y-freqs (rows 1:hy) with x zero-padding to 2W
    top = cat(x_hat[1:hy, 1:hx, :],
              x_hat[1:hy, :, :] .* false,          # (hy, W) on-device zeros
              x_hat[1:hy, W-hx+1:W, :]; dims=2)    # → (hy, 2W, C*B)

    # bot: Nyquist + new y-frequencies, all zero, width 2W
    z    = x_hat .* false                           # (hy+1, W, C*B) on-device zeros
    bot  = cat(z[:, 1:hx, :], z, z[:, W-hx+1:W, :]; dims=2)  # → (hy+1, 2W, C*B)

    x_hat_padded = cat(top, bot; dims=1)            # (H+1, 2W, C*B)
    x_up_flat    = irfft(x_hat_padded, 2 * H, 1:2) # (2H, 2W, C*B)
    return reshape(x_up_flat .* T(4), 2 * H, 2 * W, C, B)
end

# ── Upsampling block ───────────────────────────────────────────────────────────

struct UpsampleBlock{CV, PJ} <: Lux.AbstractLuxContainerLayer{(:convs, :proj)}
    convs::CV
    proj::PJ
end

function _make_conv_block(C::Int, k::Int, act, norm_type::Symbol, n_groups::Int)
    if norm_type == :batch
        return Chain(CircConv(C, C, k), BatchNorm(C), WrappedFunction(act))
    elseif norm_type == :group
        return Chain(CircConv(C, C, k), GroupNorm(C, n_groups), WrappedFunction(act))
    else
        return Chain(CircConv(C, C, k), WrappedFunction(act))
    end
end

function UpsampleBlock(C_in::Int, C_out::Int, k::Int, act,
                        norm_type::Symbol, n_groups::Int, n_convs::Int)
    conv_blocks = ntuple(_ -> _make_conv_block(C_in, k, act, norm_type, n_groups), n_convs)
    convs = Chain(conv_blocks...)
    proj  = Conv((1, 1), C_in => C_out)
    return UpsampleBlock(convs, proj)
end

function (block::UpsampleBlock)(x, ps, st)
    x_up          = spectral_upsample_2x(x)
    x_conv, st_cv = block.convs(x_up, ps.convs, st.convs)
    x_res         = x_conv .+ x_up
    x_out, st_pj  = block.proj(x_res, ps.proj, st.proj)
    return x_out, (convs=st_cv, proj=st_pj)
end

# ── ConvDecoder ────────────────────────────────────────────────────────────────

struct ConvDecoder{FC, BL, FN} <: Lux.AbstractLuxContainerLayer{(:fc, :blocks, :final_conv)}
    fc::FC
    blocks::BL
    final_conv::FN
    grid::SpectralGrid
    H0::Int
end

function ConvDecoder(latent_dim::Int, fc_hidden::Vector{Int}, init_channels::Int,
                     n_upsample_blocks::Int, conv_channels::Vector{Int},
                     n_convs_per_block::Int, kernel_size::Int,
                     act, norm_type::Symbol, n_groups::Int,
                     N::Int, rng, T::Type{<:AbstractFloat}=Float32)
    H0     = N ÷ (2^n_upsample_blocks)
    fc_out = init_channels * H0 * H0

    # FC: latent_dim → [hidden → LayerNorm]... → fc_out  (no act on final layer)
    fc_dims = [latent_dim; fc_hidden]
    n_hidden = length(fc_hidden)
    hidden_layers = ntuple(i -> Chain(Dense(fc_dims[i] => fc_dims[i+1], act),
                                      LayerNorm((fc_dims[i+1],))), n_hidden)
    fc = Chain(hidden_layers..., Dense(fc_dims[end] => fc_out))

    # Upsampling blocks: channel progression init_channels → conv_channels[1] → ...
    ch = [init_channels; conv_channels]
    block_layers = ntuple(i -> UpsampleBlock(ch[i], ch[i+1], kernel_size, act,
                                              norm_type, n_groups, n_convs_per_block),
                          n_upsample_blocks)
    blocks = Chain(block_layers...)

    # Final conv: conv_channels[end] → 1 (no activation; output is unconstrained vorticity)
    final_conv = CircConv(conv_channels[end], 1, kernel_size)

    grid  = SpectralGrid(N)
    model = ConvDecoder(fc, blocks, final_conv, grid, H0)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function (model::ConvDecoder)(x, ps, st)
    h, st_fc  = model.fc(x, ps.fc, st.fc)
    B         = size(h, 2)
    h         = reshape(h, model.H0, model.H0, :, B)
    h, st_bl  = model.blocks(h, ps.blocks, st.blocks)
    h4, st_fn = model.final_conv(h, ps.final_conv, st.final_conv)
    omega     = reshape(h4, size(h4, 1), size(h4, 2), size(h4, 4))
    return omega, (fc=st_fc, blocks=st_bl, final_conv=st_fn)
end

# ── eval_decoder_vort / eval_decoder_vel dispatches ───────────────────────────

function eval_decoder_vort(model::ConvDecoder, ::Integer, x, ps, st)
    return model(x, ps, st)
end

function eval_decoder_vel(model::ConvDecoder, N_out::Integer, x, ps, st)
    omega, st   = model(x, ps, st)
    omega_hat   = rfft(omega, 1:2)
    kx          = model.grid.kx
    ky          = model.grid.ky
    dxOp        = complex.(zero(kx), kx)
    dyOp        = complex.(zero(ky), ky)
    psi_hat     = omega_hat ./ model.grid.lap
    u = irfft(dyOp   .* psi_hat, N_out, 1:2)
    v = irfft(.-dxOp .* psi_hat, N_out, 1:2)
    return u, v, st
end
