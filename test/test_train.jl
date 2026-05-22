using Test
using Gen_DA.NN
using Lux, Lux.Training, Optimisers, Random
using Enzyme

@testset "build_optimizer" begin
    opt = NN.build_optimizer(1f-3)
    @test opt isa Optimisers.Adam
end

@testset "sinkhorn_divergence" begin
    rng = Xoshiro(42)
    d, m = 8, 20
    P = randn(rng, Float32, d, m)
    Q = randn(rng, Float32, d, m) .+ 3f0   # shifted distribution

    s_self, res_self = NN.sinkhorn_divergence(P, P, 0.1f0; n_iter=50)
    s_diff, res_diff = NN.sinkhorn_divergence(P, Q, 0.1f0; n_iter=50)

    @test s_self isa Float32
    @test abs(s_self) < 1f-2          # S(P,P) ≈ 0
    @test s_diff > 0f0                # S(P,Q) > 0 when P ≠ Q
    @test res_self isa Float32        # residual is a scalar
    @test res_diff >= 0f0
    @test res_self < 0.1f0            # should be well-converged at n_iter=50 for small d, m
end

