using Lux
using AbstractFFTs

struct VortFourierDecoder{L} <: Lux.AbstractLuxLayer
  net::L
  grid::SpectralGrid
end

function VortFourierDecoder(hidden_layers::AbstractArray{Int, 1}, num_freq::Int, rng, T::Type{<:AbstractFloat}=Float32)
  NDOF = num_freq * 2 - 1
  grid = SpectralGrid(NDOF)

  output_dim = (NDOF÷2+1) * NDOF * 2
  net = Chain(
    [Chain(Dense(dim_in => dim_out, gelu), LayerNorm((dim_out,)))
     for (dim_in, dim_out) in zip(hidden_layers[1:end-1], hidden_layers[2:end])]...,
    Dense(hidden_layers[end] => output_dim)
  )
  ps, st = Lux.setup(rng, net)
  return VortFourierDecoder(net, grid), ps, st
end

function _decode_vort_hat(model::VortFourierDecoder, x, ps, st)
  y, st = model.net(x, ps, st)
  NDOF = model.grid.N
  nfreq = NDOF ÷ 2 + 1
  batch_size = size(y, 2)
  half = nfreq * NDOF
  omega_hat_re = reshape(y[1:half, :],     nfreq, NDOF, batch_size)
  omega_hat_im = reshape(y[half+1:end, :], nfreq, NDOF, batch_size)
  return omega_hat_re, omega_hat_im, st
end

function spectral_pad(omega_hat, N_out)
  nfreq_in  = size(omega_hat, 1)
  N_in      = size(omega_hat, 2)
  trailing  = size(omega_hat)[3:end]
  nfreq_out = N_out ÷ 2 + 1
  half_in   = N_in ÷ 2
  flat      = reshape(omega_hat, nfreq_in, N_in, :)
  batch     = size(flat, 3)
  T         = eltype(omega_hat)
  # Non-mutating pad: cat slices to avoid setindex! (required for Zygote/ChainRules compat)
  low_x    = flat[1:half_in, 1:half_in, :]
  high_x   = flat[1:half_in, N_in-half_in+1:N_in, :]
  top_rows = cat(low_x, zeros(T, half_in, N_out - 2*half_in, batch), high_x; dims=2)
  padded   = cat(top_rows, zeros(T, nfreq_out - half_in, N_out, batch); dims=1)
  return reshape(padded, nfreq_out, N_out, trailing...)
end

function eval_decoder_vort(model::VortFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat_re, omega_hat_im, st = _decode_vort_hat(model, x, ps, st)
  omega_hat = complex.(omega_hat_re, omega_hat_im)
  padded    = spectral_pad(omega_hat, N_out)
  omega     = irfft(padded, N_out, 1:2)
  return omega, st
end

function eval_decoder_vel(model::VortFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat_re, omega_hat_im, st = _decode_vort_hat(model, x, ps, st)
  omega_hat = complex.(omega_hat_re, omega_hat_im)

  kx = model.grid.kx
  ky = model.grid.ky
  dxOp = complex.(zero(kx), kx)
  dyOp = complex.(zero(ky), ky)
  psi_hat = omega_hat ./ model.grid.lap

  u = irfft(spectral_pad(dyOp .* psi_hat, N_out), N_out, 1:2)
  v = irfft(spectral_pad(.-dxOp .* psi_hat, N_out), N_out, 1:2)

  return u, v, st
end
