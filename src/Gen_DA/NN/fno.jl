using Lux, Random
using AbstractFFTs

# ── FNOLayer ─────────────────────────────────────────────────────────────────
# Single FNO layer. Spectral branch applies a shared complex weight R (C_out×C_in)
# to ONLY the truncated low-frequency modes — never to the full (N_freq×N) spectrum.
# This avoids materialising the full (N_freq*N2*B) intermediate tensor through the
# matmul, which would cause OOM at training batch sizes.
#
# Truncation keeps:
#   y-modes   [1:n_modes]              (rfft is one-sided; no negative y-freqs)
#   x-modes   [1:n_modes, N-n_modes+1:N] (positive and negative low x-freqs)
#
# The full-size spectral output is reconstructed via cat with on-device zero blocks
# derived from slices of v_hat (same pattern as spectral_upsample_2x in conv_decoder).
# Requires in_chs == out_chs so that v_hat slices give zero blocks of the right depth.
# All FNOLayer instances satisfy this: interior FNO layers always have C = fno_channels.
#
# Bypass branch: 1×1 Conv in physical space. Output: gelu(spectral + bypass).

function zero_final_project!(ps_fno)
    ps_fno.project.weight .= 0f0

    if hasproperty(ps_fno.project, :bias)
        ps_fno.project.bias .= 0f0
    end

    return ps_fno
end

struct FNOLayer{C} <: Lux.AbstractLuxContainerLayer{(:bypass,)}
    bypass::C
    in_chs::Int
    out_chs::Int
    n_modes::Int
end

function FNOLayer(N::Int, n_modes::Int, in_chs::Int, out_chs::Int)
    bypass = Conv((1, 1), in_chs => out_chs; use_bias=false)
    FNOLayer(bypass, in_chs, out_chs, n_modes)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FNOLayer)
    bypass_ps = Lux.initialparameters(rng, layer.bypass)
    scale = Float32(1 / sqrt(layer.in_chs))
    R_real = scale * randn(rng, Float32, layer.out_chs, layer.in_chs)
    R_imag = zeros(Float32, layer.out_chs, layer.in_chs)
    return (bypass=bypass_ps, R_real=R_real, R_imag=R_imag)
end

# Apply complex weight R: (C_out, C_in) to a block of shape (m_y, m_x, C_in, B).
# Returns (m_y, m_x, C_out, B).
function _apply_R(R, block)
    m_y, m_x, C_in, B = size(block)
    C_out = size(R, 1)
    v_perm = permutedims(block, (3, 1, 2, 4))           # (C_in, m_y, m_x, B)
    v_2d   = reshape(v_perm, C_in, m_y * m_x * B)
    out_2d = R * v_2d                                   # (C_out, m_y*m_x*B)
    return permutedims(reshape(out_2d, C_out, m_y, m_x, B), (2, 3, 1, 4))
end

function (layer::FNOLayer)(v, ps, st)
    # v: (N, N, C_in, B)
    N1, N2, C_in, B = size(v)
    m      = layer.n_modes
    N_freq = N1 ÷ 2 + 1

    # Spectral branch ──────────────────────────────────────────────────────────
    v_hat = rfft(v, 1:2)                              # (N_freq, N2, C_in, B) complex

    # Slice low-freq blocks: (m, m, C_in, B) — keeps O(m²B) not O(N_freq*N2*B)
    v_pos = v_hat[1:m, 1:m,          :, :]            # low +x modes
    v_neg = v_hat[1:m, N2-m+1:N2,   :, :]            # low -x modes

    R       = complex.(ps.R_real, ps.R_imag)           # (C_out, C_in) complex
    out_pos = _apply_R(R, v_pos)                       # (m, m, C_out, B)
    out_neg = _apply_R(R, v_neg)                       # (m, m, C_out, B)

    # Reconstruct (N_freq, N2, C_out, B) via cat with on-device zero blocks.
    # v_hat slices .* false give correctly-typed zeros on the same device.
    mid_zeros = v_hat[1:m,          1:N2-2*m, :, :] .* false  # (m, N2-2m, C_in≡C_out, B)
    top_rows  = cat(out_pos, mid_zeros, out_neg; dims=2)        # (m, N2, C_out, B)
    bot_zeros = v_hat[1:N_freq-m,   :,        :, :] .* false  # (N_freq-m, N2, C_out, B)
    out_hat   = cat(top_rows, bot_zeros; dims=1)                # (N_freq, N2, C_out, B)

    spectral_out = irfft(out_hat, N1, 1:2)            # (N1, N2, C_out, B)

    # Bypass branch (1×1 conv, physical space) ─────────────────────────────────
    bypass_out, new_bypass_st = layer.bypass(v, ps.bypass, st.bypass)

    return gelu.(spectral_out .+ bypass_out), (bypass=new_bypass_st,)
end

# ── FourierNeuralOperator ────────────────────────────────────────────────────
# Lift (1 → fno_channels) → n_fno_layers FNOLayers → Project (fno_channels → 1),
# with a residual skip: output = FNO(input) + input.
# A single call maps (N, N, B) → (N, N, B).

struct FourierNeuralOperator{L, F, P} <: Lux.AbstractLuxContainerLayer{(:lift, :layers, :project)}
    lift::L
    layers::F
    project::P
    N::Int
    n_modes::Int
end

function FourierNeuralOperator(N::Int, n_modes::Int, fno_channels::Int, n_fno_layers::Int)
    lift       = Conv((1, 1), 1 => fno_channels; use_bias=false)
    fno_layers = ntuple(_ -> FNOLayer(N, n_modes, fno_channels, fno_channels), n_fno_layers)
    project    = Conv((1, 1), fno_channels => 1)
    FourierNeuralOperator(lift, Chain(fno_layers...), project, N, n_modes)
end

# Low-pass filter omega (N1, N2, B) to n_modes using the same frequency mask as FNOLayer.
# Keeps y-modes [1:m] and x-modes [1:m, N2-m+1:N2]; zeros the rest.
function _lowpass(omega::AbstractArray, n_modes::Int)
    N1, N2, B = size(omega)
    m      = n_modes
    N_freq = N1 ÷ 2 + 1
    omega_hat = rfft(omega, 1:2)                                        # (N_freq, N2, B)
    mid_zeros = omega_hat[1:m,        1:N2-2m,        :] .* false      # (m, N2-2m, B)
    top_row   = cat(omega_hat[1:m, 1:m, :], mid_zeros,
                    omega_hat[1:m, N2-m+1:N2, :]; dims=2)              # (m, N2, B)
    bot_zeros = omega_hat[1:N_freq-m, :,            :] .* false        # (N_freq-m, N2, B)
    omega_hat_lp = cat(top_row, bot_zeros; dims=1)                     # (N_freq, N2, B)
    return irfft(omega_hat_lp, N1, 1:2)                                # (N1, N2, B)
end

function (fno::FourierNeuralOperator)(omega, ps, st)
    # omega: (N, N, B)
    N1, N2, B = size(omega)

    # Truncate to n_modes before processing so the residual skip cannot
    # reintroduce high-frequency content the FNO spectral branch cannot represent.
    omega_lp = _lowpass(omega, fno.n_modes)                            # (N, N, B)

    v = reshape(omega_lp, N1, N2, 1, B)
    v, new_lift_st   = fno.lift(v, ps.lift, st.lift)                   # (N, N, C, B)
    v, new_layers_st = fno.layers(v, ps.layers, st.layers)             # (N, N, C, B)
    v, new_proj_st   = fno.project(v, ps.project, st.project)         # (N, N, 1, B)

    omega_out = reshape(v, N1, N2, B) .+ omega_lp                     # residual on filtered input
    return omega_out, (lift=new_lift_st, layers=new_layers_st, project=new_proj_st)
end

# ── UpsamplerWithFNO ─────────────────────────────────────────────────────────
# Combines any upsampling model (VortFourierDecoder or ConvDecoder) with a
# FourierNeuralOperator applied autoregressively for n_fno_steps iterations.
#
# ps and st are (upsampler=..., fno=...) — assembled by the constructor below,
# not via Lux.setup(rng, UpsamplerWithFNO(...)).

struct UpsamplerWithFNO{M, F} <: Lux.AbstractLuxLayer
    upsampler::M
    fno::F
    N::Int
    n_fno_steps::Int
    grid::SpectralGrid   # SpectralGrid(N) for velocity derivation at full resolution
end

function UpsamplerWithFNO(upsampler, ps_up, st_up, N::Int,
                          n_modes::Int, fno_channels::Int, n_fno_layers::Int, n_fno_steps::Int,
                          rng::AbstractRNG)
    fno            = FourierNeuralOperator(N, n_modes, fno_channels, n_fno_layers)
    ps_fno, st_fno = Lux.setup(rng, fno)
    ps_fno = zero_final_project!(ps_fno)
    model          = UpsamplerWithFNO(upsampler, fno, N, n_fno_steps, SpectralGrid(N))
    ps             = (upsampler=ps_up, fno=ps_fno)
    st             = (upsampler=st_up, fno=st_fno)
    return model, ps, st
end

function eval_decoder_vort(model::UpsamplerWithFNO, ::Integer, x, ps, st)
    omega, new_up_st = eval_decoder_vort(model.upsampler, model.N, x, ps.upsampler, st.upsampler)
    fno_st = st.fno
    for _ in 1:model.n_fno_steps
        omega, fno_st = model.fno(omega, ps.fno, fno_st)
    end
    return omega, (upsampler=new_up_st, fno=fno_st)
end

function eval_decoder_vel(model::UpsamplerWithFNO, N_out::Integer, x, ps, st)
    omega, new_st = eval_decoder_vort(model, N_out, x, ps, st)
    omega_hat     = rfft(omega, 1:2)
    psi_hat       = omega_hat ./ model.grid.lap
    u, v          = velocity_from_psi_hat(model.grid, psi_hat)
    return u, v, new_st
end
