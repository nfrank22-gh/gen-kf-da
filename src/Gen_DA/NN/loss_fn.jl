using Statistics

function sliced_wasserstein(P, Q, thetas::AbstractMatrix)
    norms = sqrt.(sum(abs2, thetas, dims=2))
    thetas_n = thetas ./ norms
    P_proj = sort(thetas_n * P, dims=2)
    Q_proj = sort(thetas_n * Q, dims=2)
    return mean(abs.(P_proj .- Q_proj))
end

function mse_reconstruction(P, Q)
    return mean((P .- Q) .^ 2)
end

# Reparameterisation sampling for the latent posterior.
# Returns (z, mu, log_sigma) — mu and log_sigma are passed to kl_regularization.
function _sample_latents(ps, cols, eps)
    mu        = ps.latent_mu[:, cols]
    log_sigma = ps.latent_log_sigma[:, cols]
    z         = mu .+ exp.(log_sigma) .* eps
    return z, mu, log_sigma
end

# KL Regularization: KL(N(mu, sigma²) || N(0,I)), mean over latent dim and batch.
function kl_regularization(mu, log_sigma)
    sigma = exp.(log_sigma)
    return 0.5f0 * mean(mean(sigma .^ 2 .+ mu .^ 2 .- 1f0 .- 2f0 .* log_sigma, dims=1))
end

function _total_kl(mu, log_sigma, kl_weight)
    return kl_weight * kl_regularization(mu, log_sigma)
end

function loss_fn(model, NDOF, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin, thetas, cols, eps, kl_weight;
    recon_loss::Symbol=:swd)
  z, mu, log_sigma = _sample_latents(ps, cols, eps)
  u, v, st    = eval_decoder_vel(model, NDOF, z, ps, st)
  u_meas      = reshape(u, size(u, 1) * size(u, 2), :)[sensor_lin, :]
  v_meas      = reshape(v, size(v, 1) * size(v, 2), :)[sensor_lin, :]
  P   = vcat(u_meas, v_meas)
  Q   = vcat(u_meas_trg, v_meas_trg)
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight), st
end

# Like loss_fn but each sample has its own sensor layout.
# sensor_lin_batch: (n_meas, batch_size) Int32 matrix of linearized indices, one column per sample.
function loss_fn_per_sample_sensors(model, N, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin_batch, thetas, cols, eps, kl_weight;
    recon_loss::Symbol=:swd)
  z, mu, log_sigma = _sample_latents(ps, cols, eps)
  u, v, st    = eval_decoder_vel(model, N, z, ps, st)
  N2          = Int32(N * N)
  B           = size(u, 3)
  u_flat      = reshape(u, N2, B)
  v_flat      = reshape(v, N2, B)
  col_offsets = reshape(Int32.(0:B-1), 1, B) .* N2
  lin_idx     = sensor_lin_batch .+ col_offsets
  u_meas      = u_flat[lin_idx]
  v_meas      = v_flat[lin_idx]
  P   = vcat(u_meas, v_meas)
  Q   = vcat(u_meas_trg, v_meas_trg)
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight), st
end

# Loss for ObservationsMode with encoder (fixed sensor layout).
# obs_features: (6, n_meas, B) — rows 1-2 are (u,v) targets; rows 3-6 are position features.
# sensor_lin: constant Vector{Int} for gathering model-predicted (u,v) at sensor locations.
function loss_fn_encoder(model, N, ps, st,
    obs_features, sensor_lin, thetas, eps, kl_weight;
    recon_loss::Symbol=:swd)
  mu, log_sigma, st = encode(model, obs_features, ps, st)
  z           = mu .+ exp.(log_sigma) .* eps
  u, v, st    = eval_decoder_vel(model, N, z, ps, st)
  u_meas      = reshape(u, size(u, 1) * size(u, 2), size(u, 3))[sensor_lin, :]
  v_meas      = reshape(v, size(v, 1) * size(v, 2), size(v, 3))[sensor_lin, :]
  u_trg       = obs_features[1, :, :]
  v_trg       = obs_features[2, :, :]
  P   = vcat(u_meas, v_meas)
  Q   = vcat(u_trg,  v_trg)
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight), st
end

# Loss for ObservationsMode with encoder and per-sample random sensor layouts.
# obs_features: (6, n_meas, B) — rows 1-2 are (u,v) targets; rows 3-6 are position features.
# sensor_lin_batch: (n_meas, B) Int32 matrix of linearised indices, one column per sample.
function loss_fn_encoder_per_sample(model, N, ps, st,
    obs_features, sensor_lin_batch, thetas, eps, kl_weight;
    recon_loss::Symbol=:swd)
  mu, log_sigma, st = encode(model, obs_features, ps, st)
  z           = mu .+ exp.(log_sigma) .* eps
  u, v, st    = eval_decoder_vel(model, N, z, ps, st)
  N2          = Int32(N * N)
  B           = size(u, 3)
  u_flat      = reshape(u, N2, B)
  v_flat      = reshape(v, N2, B)
  col_offsets = reshape(Int32.(0:B-1), 1, B) .* N2
  lin_idx     = sensor_lin_batch .+ col_offsets
  u_meas      = u_flat[lin_idx]
  v_meas      = v_flat[lin_idx]
  u_trg       = obs_features[1, :, :]
  v_trg       = obs_features[2, :, :]
  P   = vcat(u_meas, v_meas)
  Q   = vcat(u_trg,  v_trg)
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight), st
end

function loss_fn_vort_state(model, N_out::Integer, ps, st,
    omega_trg, thetas, cols, eps, kl_weight;
    recon_loss::Symbol=:swd)
  z, mu, log_sigma = _sample_latents(ps, cols, eps)
  omega, st   = eval_decoder_vort(model, N_out, z, ps, st)
  flat_dim    = N_out * N_out
  P   = reshape(omega,     flat_dim, size(omega, 3))
  Q   = reshape(omega_trg, flat_dim, size(omega_trg, 3))
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight), st
end
