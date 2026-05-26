using Lux
using AbstractFFTs

struct StreamFourierDecoder{L} <: Lux.AbstractLuxLayer
  net::L
  grid::SpectralGrid
end

function StreamFourierDecoder(hidden_layers::AbstractArray{Int, 1}, num_freq::Int, rng, T::Type{<:AbstractFloat}=Float32)
  NDOF = num_freq * 2 - 1
  grid = SpectralGrid(NDOF)

  # 2× the spectral coefficients: (psi_hat_re, psi_hat_im)
  output_dim = (NDOF÷2+1) * NDOF * 2
  net = Chain(
    [Chain(Dense(dim_in => dim_out, gelu), LayerNorm((dim_out,)))
     for (dim_in, dim_out) in zip(hidden_layers[1:end-1], hidden_layers[2:end])]...,
    Dense(hidden_layers[end] => output_dim)
  )
  ps, st = Lux.setup(rng, net)
  return StreamFourierDecoder(net, grid), ps, st
end

function _decode_omega_hat(model::StreamFourierDecoder, x, ps, st)
  y, st = model.net(x, ps, st)
  NDOF = model.grid.N
  nfreq = NDOF ÷ 2 + 1
  batch_size = size(y, 2)
  half = nfreq * NDOF
  omega_hat_re = reshape(y[1:half,    :], nfreq, NDOF, batch_size)
  omega_hat_im = reshape(y[half+1:end,:], nfreq, NDOF, batch_size)
  omega_hat = complex.(omega_hat_re, omega_hat_im) .* model.grid.dc_mask
  return omega_hat, st
end

function eval_decoder_vel(model::StreamFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat, st = _decode_omega_hat(model, x, ps, st)
  psi_hat = .-omega_hat ./ model.grid.lap
  u, v = velocity_from_psi_hat(model.grid, psi_hat, N_out)
  return u, v, st
end

function eval_decoder_vort(model::StreamFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat, st = _decode_omega_hat(model, x, ps, st)
  return irfft(spectral_pad(omega_hat, N_out), N_out, 1:2), st
end

function eval_decoder_vort_and_hat(model::StreamFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat, st = _decode_omega_hat(model, x, ps, st)
  return irfft(spectral_pad(omega_hat, N_out), N_out, 1:2), omega_hat, st
end

function eval_decoder_vel_vort_hat(model::StreamFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat, st = _decode_omega_hat(model, x, ps, st)
  psi_hat = .-omega_hat ./ model.grid.lap
  u, v    = velocity_from_psi_hat(model.grid, psi_hat, N_out)
  return u, v, omega_hat, st
end
