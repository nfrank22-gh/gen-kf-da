using Test
using Gen_DA.NN
using Random

@testset "spectral_pad: output shape" begin
    nfreq_in = 4; N_in = 7; batch = 3; N_out = 16
    omega_hat = randn(ComplexF32, nfreq_in, N_in, batch)
    padded = NN.spectral_pad(omega_hat, N_out)
    @test size(padded) == (N_out ÷ 2 + 1, N_out, batch)
end

@testset "spectral_pad: frequency window placement and zero slots" begin
    nfreq_in = 4; N_in = 7; batch = 2; N_out = 16
    # half_in governs both ky and kx truncation:
    # the highest ky row (nfreq_in) and the kx Nyquist column are dropped
    half_in = N_in ÷ 2   # = 3

    rng = MersenneTwister(1)
    omega_hat = randn(rng, ComplexF32, nfreq_in, N_in, batch)
    padded = NN.spectral_pad(omega_hat, N_out)

    # positive-kx window placed at the start
    @test padded[1:half_in, 1:half_in, :]             == omega_hat[1:half_in, 1:half_in, :]
    # negative-kx window shifted to the end of the output
    @test padded[1:half_in, N_out-half_in+1:N_out, :] == omega_hat[1:half_in, N_in-half_in+1:N_in, :]
    # interior kx slot is zero-padded (higher frequencies)
    @test all(iszero, padded[1:half_in, half_in+1:N_out-half_in, :])
    # high-ky rows are zero (includes the dropped nfreq_in row)
    @test all(iszero, padded[half_in+1:end, :, :])
end
