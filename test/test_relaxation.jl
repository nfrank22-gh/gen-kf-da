@testset "Relaxation" begin
    using Gen_DA.NN
    using Gen_DA.Solver: KfRhs
    using FFTW
    using Random

    N   = 16
    rng = Xoshiro(42)
    rhs = KfRhs(100, 4, N)

    batch_size = 3
    omega      = randn(rng, Float32, N, N, batch_size)
    omega_hat  = rfft(omega, 1:2)

    @testset "relax shape and type" begin
        relaxed = relax(rhs, omega_hat, 2, 0.01f0)
        @test size(relaxed) == size(omega_hat)
        @test eltype(relaxed) == ComplexF32
    end

    @testset "relax 0 steps is identity" begin
        relaxed = relax(rhs, omega_hat, 0, 0.01f0)
        @test relaxed === omega_hat
    end

    @testset "relax changes the field" begin
        relaxed = relax(rhs, omega_hat, 5, 0.01f0)
        @test !isapprox(Array(relaxed), Array(omega_hat))
    end

    @testset "loss_fn_relaxation forward pass" begin
        latent_dim = 8
        num_freq   = 4
        layers     = [latent_dim, 32, 64]
        model, ps, st = VortFourierDecoder(layers, num_freq, rng, Float32)

        x         = randn(rng, Float32, latent_dim, batch_size)
        omega_trg = randn(rng, Float32, N, N, batch_size)
        thetas    = randn(rng, Float32, 50, N * N)

        loss, _ = loss_fn_relaxation(model, N, x, ps, st, rhs, 2, 0.01f0, omega_trg, thetas)
        @test isfinite(loss)
        @test loss >= 0
    end
end
