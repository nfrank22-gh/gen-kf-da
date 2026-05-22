using Random: AbstractRNG

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

function CircConv(in_ch::Int, out_ch::Int, k::Int; bias::Bool=true, init_weight=kaiming_normal)
    CircConv(Conv((k, k), in_ch => out_ch; pad=0, use_bias=bias, init_weight=init_weight), k ÷ 2)
end

function (l::CircConv)(x, ps, st)
    y, new_conv_st = l.conv(_circ_pad_2d(x, l.pad), ps.conv, st.conv)
    return y, (conv=new_conv_st,)
end

# ── Hybrid local-global conv (SpectralCircConv) ────────────────────────────────
# Each layer runs two parallel branches and sums their outputs:
#   Conv branch  — CircConv with circular padding (captures local structure)
#   Spectral branch — truncated FNO-style: rfft → corner channel-mix → irfft
#                     (captures global structure via the k_max lowest modes)
#
# Weights W_lo / W_hi are stored as split re+im Float32 tensors so all Lux
# parameters remain plain Float32 (safe for Enzyme / Reactant).
# On-device complex zeros are derived from x_hat via .* false, following the
# same pattern used in spectral_upsample_2x.
#
# Constraint: k_max ≤ W÷2 where W is the spatial width at call time.
# C_in ≥ C_out must hold for the zero-deriving trick; this is satisfied for
# every site where SpectralCircConv is used (dense convs reduce channels,
# main_conv and tails preserve or reduce).

struct SpectralCircConv{C} <: Lux.AbstractLuxLayer
    conv::C      # CircConv sublayer (conv branch)
    k_max::Int
    in_ch::Int
    out_ch::Int
end

function SpectralCircConv(in_ch::Int, out_ch::Int, k::Int, k_max::Int;
                           bias::Bool=true, init_weight=kaiming_normal)
    SpectralCircConv(CircConv(in_ch, out_ch, k; bias=bias, init_weight=init_weight),
                     k_max, in_ch, out_ch)
end

function Lux.initialparameters(rng::AbstractRNG, l::SpectralCircConv)
    km    = l.k_max
    scale = Float32(1 / sqrt(l.in_ch * km * km))
    (conv    = Lux.initialparameters(rng, l.conv),
     W_lo_re = randn(rng, Float32, km, km, l.out_ch, l.in_ch) .* scale,
     W_lo_im = randn(rng, Float32, km, km, l.out_ch, l.in_ch) .* scale,
     W_hi_re = randn(rng, Float32, km, km, l.out_ch, l.in_ch) .* scale,
     W_hi_im = randn(rng, Float32, km, km, l.out_ch, l.in_ch) .* scale)
end

function Lux.initialstates(rng::AbstractRNG, l::SpectralCircConv)
    (conv=Lux.initialstates(rng, l.conv),)
end

function (l::SpectralCircConv)(x, ps, st)
    H, W, C_in, B = size(x)
    H_half = H ÷ 2 + 1
    km     = l.k_max
    C_out  = l.out_ch

    # ── Conv branch ───────────────────────────────────────────────────────────
    y_conv, new_conv_st = l.conv(x, ps.conv, st.conv)

    # ── Spectral branch ───────────────────────────────────────────────────────
    x_hat    = rfft(reshape(x, H, W, C_in * B), 1:2)   # (H_half, W, C_in*B)
    x_hat_4d = reshape(x_hat, H_half, W, C_in, B)

    x_lo = x_hat_4d[1:km, 1:km, :, :]                  # (km, km, C_in, B)
    x_hi = x_hat_4d[1:km, W-km+1:W, :, :]              # (km, km, C_in, B)

    W_lo = complex.(ps.W_lo_re, ps.W_lo_im)             # (km, km, C_out, C_in)
    W_hi = complex.(ps.W_hi_re, ps.W_hi_im)

    # Channel mix: einsum over C_in → (km, km, C_out, B)
    y_lo = dropdims(sum(reshape(W_lo, km, km, C_out, C_in, 1) .*
                        reshape(x_lo, km, km, 1, C_in, B), dims=4), dims=4)
    y_hi = dropdims(sum(reshape(W_hi, km, km, C_out, C_in, 1) .*
                        reshape(x_hi, km, km, 1, C_in, B), dims=4), dims=4)

    y_lo_f = reshape(y_lo, km, km, C_out * B)
    y_hi_f = reshape(y_hi, km, km, C_out * B)

    # On-device zeros (H_half, W, C_out*B): slice from x_hat; valid since C_in ≥ C_out
    y_hat_z = (x_hat .* false)[:, :, 1:C_out*B]

    # Scatter corners into zero-padded frequency tensor
    n_mid   = W - 2 * km
    top_row = n_mid > 0 ?
        cat(y_lo_f, y_hat_z[1:km, km+1:W-km, :], y_hi_f; dims=2) :
        cat(y_lo_f, y_hi_f; dims=2)                # (km, W, C_out*B)

    y_hat_f = cat(top_row, y_hat_z[km+1:end, :, :]; dims=1)  # (H_half, W, C_out*B)

    y_spec = reshape(irfft(y_hat_f, H, 1:2), H, W, C_out, B)

    return y_conv .+ y_spec, (conv=new_conv_st,)
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
    x_hat  = rfft(x_flat, 1:2)

    hx = W ÷ 2
    hy = H ÷ 2

    top = cat(x_hat[1:hy, 1:hx, :],
              x_hat[1:hy, :, :] .* false,
              x_hat[1:hy, W-hx+1:W, :]; dims=2)

    z   = x_hat .* false
    bot = cat(z[:, 1:hx, :], z, z[:, W-hx+1:W, :]; dims=2)

    x_hat_padded = cat(top, bot; dims=1)
    x_up_flat    = irfft(x_hat_padded, 2 * H, 1:2)
    return reshape(x_up_flat .* T(4), 2 * H, 2 * W, C, B)
end

# Iterated 2× spectral upsampling. Requires ispow2(N_out ÷ size(x, 1)).
function spectral_upsample_to(x::AbstractArray{T,4}, N_out::Int) where T
    h = x
    while size(h, 1) < N_out
        h = spectral_upsample_2x(h)
    end
    return h
end

# ── MLP builder ────────────────────────────────────────────────────────────────

function _build_mlp(in_dim::Int, hidden::Vector{Int}, out_dim::Int, act)
    dims = [in_dim; hidden]
    n_hidden = length(hidden)
    hidden_layers = ntuple(n_hidden) do i
        Chain(Dense(dims[i] => dims[i+1], act; init_weight=kaiming_normal), LayerNorm((dims[i+1],)))
    end
    return Chain(hidden_layers..., Dense(dims[end] => out_dim; init_weight=kaiming_normal))
end

# ── DenseNet Upsampling Block ──────────────────────────────────────────────────
# spectral_upsample_2x → dense block: (n_convs-1) layers of CircConv(j·C_in → C_in)
#   → InstanceNorm → act → cat, growing h from C_in to n_convs·C_in channels.
# Then: CircConv(n_convs·C_in → n_convs·C_in) → InstanceNorm → act → 1×1 proj.
# No additive skip; gradient flow is through the dense concatenation paths.
# Edge case: n_convs=1 — zero dense layers; main_conv receives C_in channels.

struct UpsampleBlock{DC, DN, CV, IN, PJ, A} <:
        Lux.AbstractLuxContainerLayer{(:dense_convs, :dense_norms, :main_conv, :main_norm, :proj)}
    dense_convs::DC   # NamedTuple{(:conv_1,...)} SpectralCircConv(j·C_in → C_in) per plain layer
    dense_norms::DN   # NamedTuple{(:norm_1,...)} InstanceNorm(C_in) per plain layer
    main_conv::CV     # SpectralCircConv: n_convs·C_in → n_convs·C_in
    main_norm::IN     # InstanceNorm(n_convs·C_in)
    proj::PJ          # Conv 1×1: n_convs·C_in → C_out
    act::A
    n_dense::Int      # n_convs - 1
end

function UpsampleBlock(C_in::Int, C_out::Int, k::Int, k_max::Int, act, n_convs::Int)
    n_dense     = n_convs - 1
    conv_keys   = ntuple(i -> Symbol(:conv_, i), n_dense)
    conv_vals   = ntuple(i -> SpectralCircConv(i * C_in, C_in, k, k_max), n_dense)
    dense_convs = NamedTuple{conv_keys}(conv_vals)
    norm_keys   = ntuple(i -> Symbol(:norm_, i), n_dense)
    norm_vals   = ntuple(_ -> InstanceNorm(C_in), n_dense)
    dense_norms = NamedTuple{norm_keys}(norm_vals)
    C_main      = n_convs * C_in
    main_conv   = SpectralCircConv(C_main, C_main, k, k_max)
    main_norm   = InstanceNorm(C_main)
    proj        = Conv((1, 1), C_main => C_out; init_weight=kaiming_normal)
    return UpsampleBlock(dense_convs, dense_norms, main_conv, main_norm, proj, act, n_dense)
end

function (block::UpsampleBlock)(x, ps, st)
    x_up = spectral_upsample_2x(x)

    h = x_up
    dense_conv_sts = st.dense_convs
    dense_norm_sts = st.dense_norms
    for i in 1:block.n_dense
        ck = Symbol(:conv_, i)
        nk = Symbol(:norm_, i)
        c, new_st_c = getfield(block.dense_convs, ck)(h, getfield(ps.dense_convs, ck), getfield(dense_conv_sts, ck))
        c, new_st_n = getfield(block.dense_norms, nk)(c, getfield(ps.dense_norms, nk), getfield(dense_norm_sts, nk))
        c = block.act.(c)
        h = cat(h, c; dims=3)
        dense_conv_sts = merge(dense_conv_sts, NamedTuple{(ck,)}((new_st_c,)))
        dense_norm_sts = merge(dense_norm_sts, NamedTuple{(nk,)}((new_st_n,)))
    end

    h_cv, st_cv = block.main_conv(h, ps.main_conv, st.main_conv)
    h_in, st_in = block.main_norm(h_cv, ps.main_norm, st.main_norm)
    h = block.act.(h_in)

    x_out, st_pj = block.proj(h, ps.proj, st.proj)
    return x_out, (dense_convs=dense_conv_sts, dense_norms=dense_norm_sts,
                   main_conv=st_cv, main_norm=st_in, proj=st_pj)
end

# ── ConvDecoder ────────────────────────────────────────────────────────────────

struct ConvDecoder{FC, BLK, TC1, TC2, A} <:
        Lux.AbstractLuxContainerLayer{(:fc, :blocks, :tail_conv1, :tail_conv2)}
    fc::FC          # z → (k_base+1)·(2·k_base)·C·2 Fourier coefficients (re+im, C channels)
    blocks::BLK     # NamedTuple{(:block_1,...)} of UpsampleBlocks
    tail_conv1::TC1 # SpectralCircConv: conv_channels[end] → 1
    tail_conv2::TC2 # SpectralCircConv: 1 → 1 (linear refinement after activation)
    act::A
    grid::SpectralGrid
    k_base::Int     # wavenumber cutoff; irfft output is (2·k_base)×(2·k_base)
    N_conv::Int     # conv backbone output resolution; spectrally upsampled to N after
    C::Int          # initial channel count (Fourier base output)
    n_blocks::Int
end

function ConvDecoder(
        latent_dim::Int,
        fc_hidden::Vector{Int},
        k_base::Int,
        C::Int,
        conv_channels::Vector{Int},
        n_convs_per_block::Int,
        kernel_sizes::Vector{Int},
        tail_kernel::Int,
        act,
        N_conv::Int,
        N::Int,
        rng,
        T::Type{<:AbstractFloat}=Float32;
        spectral_modes::Vector{Int}  = fill(4, length(conv_channels)),
        tail_spectral_modes::Int     = 4)

    n_blocks = length(conv_channels)
    @assert length(kernel_sizes) == n_blocks "kernel_sizes must have one entry per block, got $(length(kernel_sizes)) for $n_blocks blocks"
    @assert length(spectral_modes) == n_blocks "spectral_modes must have one entry per block, got $(length(spectral_modes)) for $n_blocks blocks"
    @assert 2 * k_base * (2^n_blocks) == N_conv "2·k_base·2^n_blocks must equal N_conv, got k_base=$k_base, n_blocks=$n_blocks, N_conv=$N_conv"
    @assert ispow2(N ÷ N_conv) "N ÷ N_conv must be a power of 2, got N=$N, N_conv=$N_conv"
    for i in 1:n_blocks
        spatial_i = 2 * k_base * (2^i)   # W inside block i after spectral_upsample_2x
        @assert spectral_modes[i] * 2 <= spatial_i "spectral_modes[$i]=$(spectral_modes[i]) exceeds spatial_size÷2=$(spatial_i÷2) at block $i"
    end
    @assert tail_spectral_modes * 2 <= N_conv "tail_spectral_modes=$tail_spectral_modes exceeds N_conv÷2=$(N_conv÷2)"

    # FC: latent_dim → re+im Fourier coefficients for C channels of a (k_base+1)×(2·k_base) spectrum
    fc_out = (k_base + 1) * (2 * k_base) * C * 2
    fc     = _build_mlp(latent_dim, fc_hidden, fc_out, act)

    # UpsampleBlocks: resolution doubles each step, (2·k_base)² → N_conv²
    ch         = [C; conv_channels]
    block_keys = ntuple(i -> Symbol(:block_, i), n_blocks)
    block_vals = ntuple(n_blocks) do i
        UpsampleBlock(ch[i], ch[i+1], kernel_sizes[i], spectral_modes[i], act, n_convs_per_block)
    end
    blocks = NamedTuple{block_keys}(block_vals)

    tail_conv1 = SpectralCircConv(conv_channels[end], 1, tail_kernel, tail_spectral_modes)
    tail_conv2 = SpectralCircConv(1, 1, tail_kernel, tail_spectral_modes; init_weight=glorot_uniform)

    grid  = SpectralGrid(N)
    model = ConvDecoder(fc, blocks, tail_conv1, tail_conv2, act,
                        grid, k_base, N_conv, C, n_blocks)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function _eval_psi_physical(model::ConvDecoder, x, ps, st)
    B_batch = size(x, 2)
    Hy   = model.k_base + 1
    Wx   = 2 * model.k_base
    half = Hy * Wx * model.C

    # FC → Fourier coefficients → irfft → (2·k_base)×(2·k_base)×C physical feature map
    fc_out, st_fc = model.fc(x, ps.fc, st.fc)
    re   = reshape(fc_out[1:half, :],     Hy, Wx, model.C, B_batch)
    im_  = reshape(fc_out[half+1:end, :], Hy, Wx, model.C, B_batch)
    feat_hat  = reshape(complex.(re, im_), Hy, Wx, model.C * B_batch)
    feat_phys = irfft(feat_hat, 2 * model.k_base, 1:2)
    h = reshape(feat_phys, 2 * model.k_base, 2 * model.k_base, model.C, B_batch)

    # UpsampleBlocks: each doubles resolution
    block_sts = st.blocks
    for i in 1:model.n_blocks
        blk_key = Symbol(:block_, i)
        h, new_st_i = getfield(model.blocks, blk_key)(
            h,
            getfield(ps.blocks, blk_key),
            getfield(block_sts, blk_key))
        block_sts = merge(block_sts, NamedTuple{(blk_key,)}((new_st_i,)))
    end

    # Tail: conv_channels[end] → 1 → act → 1 (linear) → ψ at N_conv×N_conv
    h1, st_tc1 = model.tail_conv1(h, ps.tail_conv1, st.tail_conv1)
    h1 = model.act.(h1)
    psi_4d, st_tc2 = model.tail_conv2(h1, ps.tail_conv2, st.tail_conv2)

    # Spectral upsample ψ from N_conv to N when N_conv < N
    if model.N_conv < model.grid.N
        psi_4d = spectral_upsample_to(psi_4d, model.grid.N)
    end
    psi = psi_4d[:, :, 1, :]   # (N, N, B_batch)

    return psi, (fc=st_fc, blocks=block_sts, tail_conv1=st_tc1, tail_conv2=st_tc2)
end

function (model::ConvDecoder)(x, ps, st)
    psi, new_st = _eval_psi_physical(model, x, ps, st)
    psi_hat = rfft(psi, 1:2) .* model.grid.dc_mask
    ikx = complex.(zero(model.grid.kx), model.grid.kx)
    iky = complex.(zero(model.grid.ky), model.grid.ky)
    u = irfft(iky .* psi_hat, model.grid.N, 1:2)
    v = irfft(.-ikx .* psi_hat, model.grid.N, 1:2)
    return u, v, new_st
end

function eval_decoder_vel(model::ConvDecoder, ::Integer, x, ps, st)
    return model(x, ps, st)
end

function eval_decoder_vort(model::ConvDecoder, ::Integer, x, ps, st)
    psi, new_st = _eval_psi_physical(model, x, ps, st)
    psi_hat   = rfft(psi, 1:2) .* model.grid.dc_mask
    omega_hat = .-model.grid.lap .* psi_hat
    return irfft(omega_hat, model.grid.N, 1:2), new_st
end
