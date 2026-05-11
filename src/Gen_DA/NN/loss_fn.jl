using Statistics

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

function loss_fn(model::VortFourierDecoder, NDOF, x, ps, st,
    u_meas_trg, v_meas_trg, sensor_lin, thetas)
  u, v, st = eval_decoder_vel(model, NDOF, x, ps, st)
  u_meas = reshape(u, size(u, 1) * size(u, 2), :)[sensor_lin, :]
  v_meas = reshape(v, size(v, 1) * size(v, 2), :)[sensor_lin, :]
  P = vcat(u_meas, v_meas)
  Q = vcat(u_meas_trg, v_meas_trg)
  return sliced_wasserstein(P, Q, thetas), st
end
