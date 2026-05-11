using Lux, Reactant, Random, AbstractFFTs
using JLD2
using FFTW
using Gen_DA.NN

Reactant.set_default_backend("cuda")
const dev = reactant_device()
const L = 2*pi

function prepare_data(Re::Real, NDOF::Int, spatial_ci::AbstractArray{<:CartesianIndex}, n_meas_time::Int, rng, T::Type{<:AbstractFloat})
  data = load("data/no_particles/Re$(Re)_N$NDOF/trajectory.jld2")
  trj_vort = T.(data["trajectory"])       # (N, N, n_saves), [y, x, t]
  t_idx = rand(rng, 1:size(trj_vort, 3), n_meas_time)

  ky_1d = 2π/L .* rfftfreq(NDOF, NDOF)
  kx_1d = 2π/L .* fftfreq(NDOF, NDOF)
  ky = T.(reshape(ky_1d, NDOF÷2+1, 1))   # (N÷2+1, 1)
  kx = T.(reshape(kx_1d, 1, NDOF))        # (1, N)

  trj_vort_hat = rfft(trj_vort, (1, 2))            # (N÷2+1, N, n_saves)

  lap = -(kx.^2 .+ ky.^2)                         # (im*kx)^2 + (im*ky)^2, shape (N÷2+1, N)
  lap[1, 1] = 1                                    # avoid div-by-zero at DC
  psi_hat = trj_vort_hat ./ lap                    # stream function: ω = ∇²ψ
  u = irfft((im .* ky) .* psi_hat, NDOF, (1, 2))  # u = ∂ψ/∂y
  v = irfft(.-(im .* kx) .* psi_hat, NDOF, (1, 2)) # v = -∂ψ/∂x

  u_t = u[:, :, t_idx]
  v_t = v[:, :, t_idx]
  u_meas = u_t[spatial_ci]
  v_meas = v_t[spatial_ci]

  return u_meas, v_meas
end

function get_meas_idx(n_meas::Int, n_meas_time::Int, NDOF::Int, rng)
  x_idx = rand(rng, 1:NDOF, n_meas, n_meas_time)
  y_idx = rand(rng, 1:NDOF, n_meas, n_meas_time)
  ci = CartesianIndex.(x_idx, y_idx, (1:n_meas_time)')
  return ci
end

function main()
  T = Float32
  Re = 100
  NDOF = 128
  n_meas_space = 20
  n_meas_time = 100

  rng = Xoshiro(123)
  spatial_ci = get_meas_idx(n_meas_space, n_meas_time, NDOF, rng)
  u_meas_trg, v_meas_trg = prepare_data(Re, NDOF, spatial_ci, n_meas_time, rng, T)

  num_freq = 16
  layers = [100, 512, 1024]
  x = randn(rng, T, layers[1], n_meas_time)

  model, ps, st = VortFourierDecoder(layers, num_freq, L, rng, T)
  
  n_slices = 50
  thetas = randn(rng, T, n_slices, 2*n_meas_space)

  if false
    ps = ps |> dev
    st = st |> dev
    x = x |> dev
    thetas = thetas |> dev
    loss_compiled = @compile loss_fn(model, NDOF, x, ps, st, u_meas_trg, v_meas_trg, spatial_ci, thetas)
    loss, st = loss_compiled(model, NDOF, x, ps, st, u_meas_trg, v_meas_trg, spatial_ci, thetas)
  else
    loss, st = loss_fn(model, NDOF, x, ps, st, u_meas_trg, v_meas_trg, spatial_ci, thetas)
  end

  println("loss: ", loss)

  return nothing
end

main()
