using Statistics

function sliced_wasserstein(P, Q, thetas::AbstractMatrix)
    norms = sqrt.(sum(abs2, thetas, dims=2))
    thetas_n = thetas ./ norms
    P_proj = sort(thetas_n * P, dims=2)
    Q_proj = sort(thetas_n * Q, dims=2)
    return mean(abs.(P_proj .- Q_proj))
end

# Analytic subgradient of sliced_wasserstein w.r.t. P.
# Uses the sorting trick: rank-order P and Q projections, then assign sign(P_sorted - Q_sorted)
# back to P's original positions. P, Q: (flat_dim, batch); thetas: (n_slices, flat_dim).
# Returns d_P of the same shape as P.
function sliced_wasserstein_adjoint(P::Array{Float32,2}, Q::Array{Float32,2}, thetas::Array{Float32,2})
    norms    = sqrt.(sum(abs2, thetas; dims=2))
    thetas_n = thetas ./ norms            # (n_slices, flat_dim)
    P_proj   = thetas_n * P               # (n_slices, batch)
    Q_proj   = thetas_n * Q               # (n_slices, batch)
    n_slices, batch = size(P_proj)
    n_total  = Float32(n_slices * batch)
    G        = zeros(Float32, n_slices, batch)
    for j in 1:n_slices
        perm_P = sortperm(P_proj[j, :])
        perm_Q = sortperm(Q_proj[j, :])
        d_sign = sign.(P_proj[j, perm_P] .- Q_proj[j, perm_Q])
        G[j, perm_P] = d_sign
    end
    return (thetas_n' * G) ./ n_total     # (flat_dim, batch)
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
