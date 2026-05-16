using Statistics

function sliced_wasserstein(P, Q, thetas::AbstractMatrix)
    norms = sqrt.(sum(abs2, thetas, dims=2))
    thetas_n = thetas ./ norms
    P_proj = sort(thetas_n * P, dims=2)
    Q_proj = sort(thetas_n * Q, dims=2)
    return mean(abs.(P_proj .- Q_proj))
end

function loss_fn(model, NDOF, x, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin, thetas)
  u, v, st = eval_decoder_vel(model, NDOF, x, ps, st)
  u_meas = reshape(u, size(u, 1) * size(u, 2), :)[sensor_lin, :]
  v_meas = reshape(v, size(v, 1) * size(v, 2), :)[sensor_lin, :]
  P = vcat(u_meas, v_meas)
  Q = vcat(u_meas_trg, v_meas_trg)
  return sliced_wasserstein(P, Q, thetas), st
end

function loss_fn_vort_state(model, N_out::Integer, x, ps, st,
    omega_trg, thetas)
  omega, st = eval_decoder_vort(model, N_out, x, ps, st)
  flat_dim = N_out * N_out
  P = reshape(omega,     flat_dim, size(omega, 3))
  Q = reshape(omega_trg, flat_dim, size(omega_trg, 3))
  return sliced_wasserstein(P, Q, thetas), st
end
