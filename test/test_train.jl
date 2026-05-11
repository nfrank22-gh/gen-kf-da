using Test
using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Lux, Lux.Training, Optimisers, Random
using Zygote  # CPU-compatible AD (Enzyme can't diff through FFTW plan creation)

@testset "build_optimizer: constant LR" begin
    opt, sched = NN.build_optimizer(1f-3, 100, false)
    @test opt isa Optimisers.Adam
    @test sched === nothing
end

@testset "build_optimizer: cosine schedule values" begin
    lr = 1f-3
    _, sched = NN.build_optimizer(lr, 100, true)
    @test sched !== nothing
    @test sched(1)   ≈ lr              # epoch 1 → full lr
    @test sched(51)  ≈ lr / 2  atol=1f-7   # midpoint → lr/2
    @test sched(51)  < sched(1)        # strictly less than max
    @test sched(101) ≈ 0f0    atol=1f-7   # end → 0
end

@testset "Training step reduces loss on CPU" begin
    T = Float32
    rng = MersenneTwister(42)

    latent_dim = 4; n_steps = 20; batch_size = 4
    # [latent_dim → 16 → output_dim] — output_dim set by num_freq via VortFourierDecoder
    model, ps, st = NN.VortFourierDecoder([latent_dim, 16], 4, T(2π), rng, T)

    opt, _ = NN.build_optimizer(1f-3, n_steps, false)
    tstate = Training.TrainState(model, ps, st, opt)

    # Dense-only MSE loss: bypasses FFT/spectral ops that have in-place mutations
    # Zygote can't trace. Tests TrainState + gradient + Adam update mechanism.
    # Production uses SWD + AutoEnzyme via Reactant/XLA (handles mutations natively).
    x0       = randn(rng, T, latent_dim, batch_size)
    y_ref, _ = Lux.apply(model.net, x0, ps, st)
    y_target = y_ref .+ T(0.5) .* randn(rng, T, size(y_ref))  # fixed noisy target

    function train_loss(m, params, states, data)
        x, y_t = data
        y, new_st = m.net(x, params, states)
        l = sum((y .- y_t).^2) / length(y)
        return l, new_st, (;)
    end

    losses = T[]
    for _ in 1:n_steps
        x = randn(rng, T, latent_dim, batch_size)
        _, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), train_loss, (x, y_target), tstate)
        push!(losses, loss)
    end

    @test all(isfinite, losses)
    @test all(>=(0), losses)
    @test last(losses) < first(losses)
end
