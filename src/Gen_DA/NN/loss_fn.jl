using Statistics
using Random

function sliced_wasserstein(P, Q, thetas::AbstractMatrix)
    norms = sqrt.(sum(abs2, thetas, dims=2))
    thetas_n = thetas ./ norms
    P_proj = sort(thetas_n * P, dims=2)
    Q_proj = sort(thetas_n * Q, dims=2)
    return mean(abs.(P_proj .- Q_proj))
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

function loss_fn_vort(model::VortFourierDecoder, x, ps, st,
    oh_re_trg, oh_im_trg, thetas)
  oh_re, oh_im, st = _decode_vort_hat(model, x, ps, st)
  nfreq, NDOF = size(oh_re, 1), size(oh_re, 2)
  flat_dim = 2 * nfreq * NDOF
  P = reshape(vcat(oh_re, oh_im), flat_dim, size(oh_re, 3))
  Q = reshape(vcat(oh_re_trg, oh_im_trg), flat_dim, size(oh_re_trg, 3))
  return sliced_wasserstein(P, Q, thetas), st
end

function loss_fn_vort_state(model::VortFourierDecoder, N_out::Integer, x, ps, st,
    omega_trg, thetas)
  omega, st = eval_decoder_vort(model, N_out, x, ps, st)
  flat_dim = N_out * N_out
  P = reshape(omega,     flat_dim, size(omega, 3))
  Q = reshape(omega_trg, flat_dim, size(omega_trg, 3))
  return sliced_wasserstein(P, Q, thetas), st
end
