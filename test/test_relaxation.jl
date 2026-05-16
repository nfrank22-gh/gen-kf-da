using Test
import Gen_DA
using Gen_DA.NN
using Gen_DA.Solver: KfRhs
using FFTW
using Random

@testset "Relaxation" begin
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

    @testset "RelaxedDecoder shape, type, and differs from raw decoder" begin
        latent_dim = 8
        num_freq   = 4
        layers     = [latent_dim, 32, 64]
        model, ps, st = VortFourierDecoder(layers, num_freq, rng, Float32)

        x  = randn(rng, Float32, latent_dim, batch_size)
        rd = RelaxedDecoder(model, rhs, 3, 0.01f0)

        omega_relaxed, _ = rd(N, x, ps, st)
        @test size(omega_relaxed) == (N, N, batch_size)
        @test eltype(omega_relaxed) == Float32
        @test all(isfinite, omega_relaxed)

        # relaxed output should differ from raw decoder output
        omega_raw, _ = eval_decoder_vort(model, N, x, ps, st)
        @test !isapprox(omega_relaxed, omega_raw)
    end

    @testset "relax_and_store shape" begin
        n_steps = 3
        traj = relax_and_store(rhs, omega_hat, n_steps, 0.01f0)
        N_freq = N ÷ 2 + 1
        @test size(traj) == (N_freq, N, batch_size, n_steps + 1)
        @test eltype(traj) == ComplexF32
        @test isapprox(traj[:, :, :, 1], omega_hat)
        @test !isapprox(traj[:, :, :, end], omega_hat)
    end

    @testset "relax_and_store consistent with relax" begin
        n_steps = 4
        traj      = relax_and_store(rhs, omega_hat, n_steps, 0.01f0)
        relaxed   = relax(rhs, omega_hat, n_steps, 0.01f0)
        @test isapprox(traj[:, :, :, end], relaxed; rtol=1f-5)
    end

    @testset "relax_adj_full output shape and finite" begin
        n_steps       = 3
        traj          = relax_and_store(rhs, omega_hat, n_steps, 0.01f0)
        d_omega_relax = randn(rng, Float32, N, N, batch_size)
        d_omega_gen   = relax_adj_full(rhs, traj, n_steps, 0.01f0, d_omega_relax, N)
        @test size(d_omega_gen) == (N, N, batch_size)
        @test eltype(d_omega_gen) == Float32
        @test all(isfinite, d_omega_gen)
    end

    @testset "relax_adj_full adjoint identity: ⟨Jv, w⟩ ≈ ⟨v, J^T w⟩" begin
        # J = d(omega_relaxed)/d(omega_gen) over n_steps of the full chain
        # omega_gen → rfft → relax → irfft → omega_relaxed
        n_steps = 2
        dt      = 0.01f0

        omega_0 = randn(rng, Float32, N, N, batch_size)
        v_gen   = randn(rng, Float32, N, N, batch_size)   # perturbation
        eps     = 1f-3

        # Jv via centred finite differences
        traj_p = relax_and_store(rhs, rfft(omega_0 .+ eps .* v_gen, 1:2), n_steps, dt)
        traj_m = relax_and_store(rhs, rfft(omega_0 .- eps .* v_gen, 1:2), n_steps, dt)
        Jv = (irfft(traj_p[:,:,:,end], N, 1:2) .- irfft(traj_m[:,:,:,end], N, 1:2)) ./ (2f0 * eps)

        w   = randn(rng, Float32, N, N, batch_size)   # co-vector
        lhs = sum(Jv .* w)                             # ⟨Jv, w⟩

        traj_0  = relax_and_store(rhs, rfft(omega_0, 1:2), n_steps, dt)
        d_gen   = relax_adj_full(rhs, traj_0, n_steps, dt, w, N)
        rhs_val = sum(v_gen .* d_gen)                  # ⟨v, J^T w⟩

        @test isapprox(lhs, rhs_val; rtol=2f-2)
    end

    @testset "sliced_wasserstein_adjoint finite-difference check" begin
        flat_dim = N * N
        n_slices = 20
        P = randn(rng, Float32, flat_dim, batch_size)
        Q = randn(rng, Float32, flat_dim, batch_size)
        thetas = randn(rng, Float32, n_slices, flat_dim)

        d_P = sliced_wasserstein_adjoint(P, Q, thetas)
        @test size(d_P) == size(P)

        v   = randn(rng, Float32, flat_dim, batch_size)
        eps = 1f-3
        swd_p = sliced_wasserstein(P .+ eps .* v, Q, thetas)
        swd_m = sliced_wasserstein(P .- eps .* v, Q, thetas)
        fd    = Float32((swd_p - swd_m) / (2 * eps))
        ad    = sum(d_P .* v)
        @test isapprox(fd, ad; rtol=5f-2)   # SWD subgradient; loose tolerance
    end
end
