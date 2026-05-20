using Lux
using Statistics

# ── Position encoding ──────────────────────────────────────────────────────────
# Convert linearised grid indices (column-major, 1-indexed in an N×N grid) to
# (sin_x, cos_x, sin_y, cos_y) Fourier features encoding the 2π-periodic torus.

function _sensor_fourier_features(sensor_lin::AbstractVector{<:Integer}, N::Int)
    rows_0   = (sensor_lin .- 1) .% N     # 0-indexed row (first dim)
    cols_0   = (sensor_lin .- 1) .÷ N     # 0-indexed col (second dim)
    x = Float32(2π) .* Float32.(cols_0) ./ Float32(N)
    y = Float32(2π) .* Float32.(rows_0) ./ Float32(N)
    return vcat(sin.(x)', cos.(x)', sin.(y)', cos.(y)')   # (4, n_meas)
end

# Per-sample variant: sensor_lin_batch is (n_meas, B). Returns (4, n_meas, B).
function _sensor_fourier_features_per_sample(
    sensor_lin_batch::AbstractMatrix{Int32}, N::Int
)
    n_meas, B = size(sensor_lin_batch)
    rows_0   = (sensor_lin_batch .- Int32(1)) .% Int32(N)
    cols_0   = (sensor_lin_batch .- Int32(1)) .÷ Int32(N)
    x = Float32(2π) .* Float32.(cols_0) ./ Float32(N)
    y = Float32(2π) .* Float32.(rows_0) ./ Float32(N)
    return cat(reshape(sin.(x), 1, n_meas, B),
               reshape(cos.(x), 1, n_meas, B),
               reshape(sin.(y), 1, n_meas, B),
               reshape(cos.(y), 1, n_meas, B); dims=1)   # (4, n_meas, B)
end

# ── DeepSetsEncoder ────────────────────────────────────────────────────────────
# Maps a (6, n_meas, B) observation tensor → (latent_dim, B) mu and log_sigma.
#   phi: shared per-sensor sub-MLP  (6 → encoder_hidden)
#   rho: aggregation head MLP       (encoder_hidden[end] → 2·latent_dim)
# Mean pooling across sensors keeps output magnitude independent of n_meas.

struct DeepSetsEncoder{P, R} <: Lux.AbstractLuxContainerLayer{(:phi, :rho)}
    phi::P
    rho::R
    latent_dim::Int
end

function DeepSetsEncoder(
    encoder_hidden::AbstractVector{Int},
    encoder_head_hidden::AbstractVector{Int},
    latent_dim::Int, rng, T::Type{<:AbstractFloat}=Float32
)
    phi_dims = [6; encoder_hidden]
    phi = Chain(
        [Chain(Dense(phi_dims[i] => phi_dims[i+1], gelu), LayerNorm((phi_dims[i+1],)))
         for i in 1:length(phi_dims)-1]...
    )

    rho_dims = [encoder_hidden[end]; encoder_head_hidden; 2 * latent_dim]
    rho = Chain(
        [Chain(Dense(rho_dims[i] => rho_dims[i+1], gelu), LayerNorm((rho_dims[i+1],)))
         for i in 1:length(rho_dims)-2]...,
        Dense(rho_dims[end-1] => rho_dims[end])
    )

    model = DeepSetsEncoder(phi, rho, latent_dim)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function (l::DeepSetsEncoder)(obs::AbstractArray{T,3}, ps, st) where T
    d, n_meas, B = size(obs)
    flat = reshape(obs, d, n_meas * B)
    embedded, new_phi_st = l.phi(flat, ps.phi, st.phi)
    h = size(embedded, 1)
    pooled = dropdims(mean(reshape(embedded, h, n_meas, B); dims=2); dims=2)  # (h, B)
    out, new_rho_st = l.rho(pooled, ps.rho, st.rho)                          # (2·ld, B)
    mu        = out[1:l.latent_dim, :]
    log_sigma = out[l.latent_dim+1:end, :]
    return mu, log_sigma, (phi=new_phi_st, rho=new_rho_st)
end

# ── ObservationEncoderDecoder ──────────────────────────────────────────────────
# Lux container pairing a DeepSetsEncoder with any upsampling model.
# Lux auto-splits ps/st: ps.encoder / ps.decoder, st.encoder / st.decoder.

struct ObservationEncoderDecoder{E, D} <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
end

# Unconditional decode path: takes z directly, bypasses encoder. Used for eval.
function eval_decoder_vel(m::ObservationEncoderDecoder, N_out::Integer, x, ps, st)
    u, v, new_dec_st = eval_decoder_vel(m.decoder, N_out, x, ps.decoder, st.decoder)
    return u, v, merge(st, (decoder=new_dec_st,))
end

function eval_decoder_vort(m::ObservationEncoderDecoder, N_out::Integer, x, ps, st)
    omega, new_dec_st = eval_decoder_vort(m.decoder, N_out, x, ps.decoder, st.decoder)
    return omega, merge(st, (decoder=new_dec_st,))
end

# Amortised encode path: observations → (mu, log_sigma). Used during training.
function encode(m::ObservationEncoderDecoder, obs_features, ps, st)
    mu, log_sigma, new_enc_st = m.encoder(obs_features, ps.encoder, st.encoder)
    return mu, log_sigma, merge(st, (encoder=new_enc_st,))
end
