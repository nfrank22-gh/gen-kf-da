using Test
using Gen_DA.NN
using Random

@testset "sliced_wasserstein_spectral: returns Float32 scalar" begin
    rng = MersenneTwister(1)
    nfreq, NDOF, batch = 5, 8, 4
    P_hat = randn(rng, ComplexF32, nfreq, NDOF, batch)
    Q_hat = randn(rng, ComplexF32, nfreq, NDOF, batch)
    result = NN.sliced_wasserstein_spectral(P_hat, Q_hat, 10; rng=MersenneTwister(0))
    @test result isa Float32
    @test ndims(result) == 0 || result isa Number
end

@testset "sliced_wasserstein_spectral: self-distance is near-zero" begin
    rng = MersenneTwister(2)
    P_hat = randn(rng, ComplexF32, 5, 8, 4)
    result = NN.sliced_wasserstein_spectral(P_hat, P_hat, 20; rng=MersenneTwister(0))
    @test result < 1f-5
end

@testset "sliced_wasserstein_spectral: non-negative for distinct inputs" begin
    rng = MersenneTwister(3)
    P_hat = randn(rng, ComplexF32, 5, 8, 4)
    Q_hat = randn(rng, ComplexF32, 5, 8, 4)
    result = NN.sliced_wasserstein_spectral(P_hat, Q_hat, 20; rng=MersenneTwister(0))
    @test result >= 0f0
end
