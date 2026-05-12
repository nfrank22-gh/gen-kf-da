using Test
using Gen_DA.NN
using Random

@testset "sliced_wasserstein_spectral: self-distance is near-zero" begin
    rng = MersenneTwister(2)
    P_hat = randn(rng, ComplexF32, 5, 8, 4)
    result = NN.sliced_wasserstein_spectral(P_hat, P_hat, 20; rng=MersenneTwister(0))
    @test result < 1f-5
end

@testset "sliced_wasserstein_spectral: converges to analytical value with increasing slices" begin
    # nfreq=1, NDOF=1 → flat_dim=2: P = delta at [1,0], Q = delta at [0,0] in R²
    # Projection θ=(cos α, sin α) on S¹ gives W1 = |cos α| per slice.
    # Analytical SWD = E_{α~Uniform(S¹)}[|cos α|] = 2/π.
    # Batch=1 eliminates sampling noise; the only error source is MC over projections.
    P_hat = ones(ComplexF32, 1, 1, 1)
    Q_hat = zeros(ComplexF32, 1, 1, 1)
    analytical = 2f0 / Float32(π)

    err(ns) = abs(NN.sliced_wasserstein_spectral(P_hat, Q_hat, ns; rng=MersenneTwister(ns)) - analytical)
    e10, e1000 = err(10), err(1000)

    @test e10 > e1000       # coarser estimate farther from truth
    @test e1000 < 0.02f0    # 1000 slices: within ~3% of analytical value
end
