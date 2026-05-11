using Lux
using AbstractFFTs

struct VortFourierDecoder{L, T, A<:AbstractArray{T, 2}} <: Lux.AbstractLuxLayer
  net::L 
  kx::A
  ky::A
end 

function VortFourierDecoder(hidden_layers::AbstractArray{Int, 1}, num_freq::Int, L::Number, rng, T::Type{<:AbstractFloat}=Float32)
  NDOF = num_freq * 2 - 1
  ky_1d = 2π/L .* rfftfreq(NDOF, NDOF)
  kx_1d = 2π/L .* fftfreq(NDOF, NDOF)
  ky = T.(reshape(ky_1d, NDOF÷2+1, 1))
  kx = T.(reshape(kx_1d, 1, NDOF))

  output_dim = (NDOF÷2+1) * NDOF * 2
  net = Chain(
    [Dense(dim_in => dim_out, gelu) for (dim_in, dim_out) in zip(hidden_layers[1:end-1], hidden_layers[2:end])]...,
    Dense(hidden_layers[end] => output_dim)
  )
  ps, st = Lux.setup(rng, net)
  return VortFourierDecoder(net, kx, ky), ps, st
end

function eval_decoder_vort_hat(model::VortFourierDecoder, x, ps, st)
  y, st = model.net(x, ps, st)
  NDOF = size(model.kx, 2)
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
  padded    = zeros(eltype(omega_hat), nfreq_out, N_out, batch)
  padded[1:half_in, 1:half_in, :]             = flat[1:half_in, 1:half_in, :]
  padded[1:half_in, N_out-half_in+1:N_out, :] = flat[1:half_in, N_in-half_in+1:N_in, :]
  return reshape(padded, nfreq_out, N_out, trailing...)
end

function eval_decoder_vort(model::VortFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat_re, omega_hat_im, st = eval_decoder_vort_hat(model, x, ps, st)
  omega_hat = complex.(omega_hat_re, omega_hat_im)
  padded    = spectral_pad(omega_hat, N_out)
  omega     = irfft(padded, N_out, 1:2)
  return omega, st
end

function eval_decoder_vel(model::VortFourierDecoder, N_out::Integer, x, ps, st)
  omega_hat_re, omega_hat_im, st = eval_decoder_vort_hat(model, x, ps, st)
  omega_hat = complex.(omega_hat_re, omega_hat_im)

  kx = model.kx  # (1, NDOF_model), broadcasts over batch dim
  ky = model.ky  # (NDOF_model÷2+1, 1)
  dxOp = complex.(zero(kx), kx)   # i*kx as ComplexF32, avoids Complex{Bool}
  dyOp = complex.(zero(ky), ky)   # i*ky as ComplexF32
  lap = -(kx.^2 .+ ky.^2)
  lap[1, 1] = 1
  psi_hat = omega_hat ./ lap

  u = irfft(spectral_pad(dyOp .* psi_hat, N_out), N_out, 1:2)
  v = irfft(spectral_pad(.-dxOp .* psi_hat, N_out), N_out, 1:2)

  return u, v, st
end
