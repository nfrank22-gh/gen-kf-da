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

# ---------------------------------------------------------------------------
# Debiased Sinkhorn divergence
# S_ε(P,Q) = OT_ε(P,Q) − ½·OT_ε(P,P) − ½·OT_ε(Q,Q)
# Ground metric: squared L2.  Cost normalised by mean(C_PQ) so ε is scale-invariant.
# Fixed-iteration log-domain Sinkhorn for XLA compatibility.
# ---------------------------------------------------------------------------

function _logsumexp(A::AbstractMatrix; dims::Int)
    m_val = maximum(A; dims=dims)
    return m_val .+ log.(sum(exp.(A .- m_val); dims=dims))
end

function _sq_cost(P::AbstractMatrix, Q::AbstractMatrix)
    P_sq = sum(abs2, P; dims=1)          # (1, m)
    Q_sq = sum(abs2, Q; dims=1)          # (1, n)
    C    = P_sq' .+ Q_sq .- 2f0 .* (P' * Q)   # (m, n)
    return max.(C, 0f0)
end

function _cosine_cost(P::AbstractMatrix, Q::AbstractMatrix)
    P_norms = sqrt.(sum(abs2, P; dims=1))   # (1, m)
    Q_norms = sqrt.(sum(abs2, Q; dims=1))   # (1, n)
    C       = 1f0 .- (P' * Q) ./ (P_norms' .* Q_norms)
    return max.(C, 0f0)
end

function _lr_mahalanobis_cost(P::AbstractMatrix, Q::AbstractMatrix, L::AbstractMatrix)
    return _sq_cost(L' * P, L' * Q)
end

function _sinkhorn_ot(C::AbstractMatrix, eps::Real, n_iter::Int)
    m, n      = size(C)
    log_m     = log(Float32(m))
    log_n     = log(Float32(n))
    f         = C[:, 1] .* 0f0       # m-vector, same device/type as C
    g         = C[1, :] .* 0f0       # n-vector
    neg_C_eps = -(C ./ eps)
    for _ in 1:n_iter
        A = f ./ eps .+ neg_C_eps                                    # (m, n)
        g = eps .* (log_m .- vec(_logsumexp(A; dims=1)))             # n-vector
        B = reshape(g, 1, :) ./ eps .+ neg_C_eps                    # (m, n)
        f = eps .* (log_n .- vec(_logsumexp(B; dims=2)))             # m-vector
    end
    # Dual residual: one extra half-step measures how far g is from its fixed point.
    A_check  = f ./ eps .+ neg_C_eps
    g_check  = eps .* (log_m .- vec(_logsumexp(A_check; dims=1)))
    residual = maximum(abs.(g_check .- g))
    return mean(f) + mean(g), residual
end

# Returns (divergence, dual_residual) where residual is from the P↔Q solve.
# metric: :sq_l2 (squared Euclidean), :cosine (cosine dissimilarity 1 − cos θ),
#         :lr_mahalanobis (squared L2 in the column space of mahalanobis_L), or
#         :rand_lr_mahalanobis (same projection formula; caller supplies a fresh
#         random mahalanobis_L each epoch).
function sinkhorn_divergence(P::AbstractMatrix, Q::AbstractMatrix, eps::Real;
                             n_iter::Int=100, metric::Symbol=:sq_l2,
                             mahalanobis_L::Union{Nothing,AbstractMatrix}=nothing)
    cost_fn = if metric == :cosine
        _cosine_cost
    elseif metric == :lr_mahalanobis || metric == :rand_lr_mahalanobis
        (p, q) -> _lr_mahalanobis_cost(p, q, mahalanobis_L)
    else
        _sq_cost
    end
    C_PQ            = cost_fn(P, Q)
    scale           = mean(C_PQ)
    ot_PQ, residual = _sinkhorn_ot(C_PQ ./ scale,            eps, n_iter)
    ot_PP, _        = _sinkhorn_ot(cost_fn(P, P) ./ scale,   eps, n_iter)
    ot_QQ, _        = _sinkhorn_ot(cost_fn(Q, Q) ./ scale,   eps, n_iter)
    return ot_PQ - 0.5f0 * (ot_PP + ot_QQ), residual
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

# H2 Vorticity Regularization: mean(|k|⁴ |ω̂|²) over modes and batch.
# omega_hat is at the model's native spectral resolution; model.grid.lap = -(kx²+ky²).
function _h2_vorticity(model, omega_hat, h2_weight)
    return h2_weight * mean(abs.(model.grid.lap) .* abs2.(omega_hat))
end

function loss_fn(model, NDOF, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin, thetas, cols, eps, kl_weight;
    h2_weight::Float32=0f0,
    recon_loss::Symbol=:swd,
    sinkhorn_eps::Float32=0.1f0,
    sinkhorn_n_iter::Int=100,
    sinkhorn_metric::Symbol=:sq_l2,
    mahalanobis_L::Union{Nothing,AbstractMatrix}=nothing)
  z, mu, log_sigma       = _sample_latents(ps, cols, eps)
  u, v, omega_hat, st    = eval_decoder_vel_vort_hat(model, NDOF, z, ps, st)
  u_meas      = reshape(u, size(u, 1) * size(u, 2), :)[sensor_lin, :]
  v_meas      = reshape(v, size(v, 1) * size(v, 2), :)[sensor_lin, :]
  P   = vcat(u_meas, v_meas)
  Q   = vcat(u_meas_trg, v_meas_trg)
  h2  = _h2_vorticity(model, omega_hat, h2_weight)
  if recon_loss == :sinkhorn
      rec, sinkhorn_res = sinkhorn_divergence(P, Q, sinkhorn_eps;
                              n_iter=sinkhorn_n_iter, metric=sinkhorn_metric,
                              mahalanobis_L=mahalanobis_L)
      return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; sinkhorn_residual=sinkhorn_res, h2_reg=h2)
  end
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; h2_reg=h2)
end

# Like loss_fn but each sample has its own sensor layout.
# sensor_lin_batch: (n_meas, batch_size) Int32 matrix of linearized indices, one column per sample.
function loss_fn_per_sample_sensors(model, N, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin_batch, thetas, cols, eps, kl_weight;
    h2_weight::Float32=0f0,
    recon_loss::Symbol=:swd,
    sinkhorn_eps::Float32=0.1f0,
    sinkhorn_n_iter::Int=100,
    sinkhorn_metric::Symbol=:sq_l2)
  z, mu, log_sigma       = _sample_latents(ps, cols, eps)
  u, v, omega_hat, st    = eval_decoder_vel_vort_hat(model, N, z, ps, st)
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
  h2  = _h2_vorticity(model, omega_hat, h2_weight)
  if recon_loss == :sinkhorn
      rec, sinkhorn_res = sinkhorn_divergence(P, Q, sinkhorn_eps; n_iter=sinkhorn_n_iter, metric=sinkhorn_metric)
      return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; sinkhorn_residual=sinkhorn_res, h2_reg=h2)
  end
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; h2_reg=h2)
end

function loss_fn_vort_state(model, N_out::Integer, ps, st,
    omega_trg, thetas, cols, eps, kl_weight;
    h2_weight::Float32=0f0,
    recon_loss::Symbol=:swd,
    sinkhorn_eps::Float32=0.1f0,
    sinkhorn_n_iter::Int=100,
    sinkhorn_metric::Symbol=:sq_l2,
    mahalanobis_L::Union{Nothing,AbstractMatrix}=nothing)
  z, mu, log_sigma           = _sample_latents(ps, cols, eps)
  omega, omega_hat, st       = eval_decoder_vort_and_hat(model, N_out, z, ps, st)
  flat_dim    = N_out * N_out
  P   = reshape(omega,     flat_dim, size(omega, 3))
  Q   = reshape(omega_trg, flat_dim, size(omega_trg, 3))
  h2  = _h2_vorticity(model, omega_hat, h2_weight)
  if recon_loss == :sinkhorn
      rec, sinkhorn_res = sinkhorn_divergence(P, Q, sinkhorn_eps;
                              n_iter=sinkhorn_n_iter, metric=sinkhorn_metric,
                              mahalanobis_L=mahalanobis_L)
      return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; sinkhorn_residual=sinkhorn_res, h2_reg=h2)
  end
  rec = recon_loss == :mse ? mse_reconstruction(P, Q) : sliced_wasserstein(P, Q, thetas)
  return rec + _total_kl(mu, log_sigma, kl_weight) + h2, st, (; h2_reg=h2)
end
