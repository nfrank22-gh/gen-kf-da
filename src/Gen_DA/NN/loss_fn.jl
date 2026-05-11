using Statistics
using Random

function sliced_wasserstein(P, Q, thetas::AbstractMatrix)
  n_slices = size(thetas, 1)
  T = eltype(P)
  loss = zero(T)
  for i in 1:n_slices
    θ = thetas[i, :]
    θ = θ ./ sqrt(sum(abs2, θ))
    loss += mean(abs.(sort(θ' * P, dims=2) .- sort(θ' * Q, dims=2)))
  end
  return loss / n_slices
end

function sliced_wasserstein_spectral(P_hat, Q_hat, n_slices::Int; rng=default_rng())
    T = real(eltype(P_hat))
    nfreq, NDOF = size(P_hat, 1), size(P_hat, 2)
    flat_dim = 2 * nfreq * NDOF
    P_flat = reshape(vcat(real.(P_hat), imag.(P_hat)), flat_dim, size(P_hat, 3))
    Q_flat = reshape(vcat(real.(Q_hat), imag.(Q_hat)), flat_dim, size(Q_hat, 3))
    thetas = randn(rng, T, n_slices, flat_dim)
    return sliced_wasserstein(P_flat, Q_flat, thetas)
end

function loss_fn(model::VortFourierDecoder, NDOF, x, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin, thetas)
  u, v, st = eval_decoder_vel(model, NDOF, x, ps, st)
  u_meas = reshape(u, size(u, 1) * size(u, 2), :)[sensor_lin, :]
  v_meas = reshape(v, size(v, 1) * size(v, 2), :)[sensor_lin, :]
  P = vcat(u_meas, v_meas)
  Q = vcat(u_meas_trg, v_meas_trg)
  return sliced_wasserstein(P, Q, thetas), st
end
