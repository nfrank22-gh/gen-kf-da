import Reactant
import Lux
import Lux.Training as Training
import Optimisers

# Immutable hyperparameters — fixed for the lifetime of a training run.
struct TrainingConfig
    n_epochs::Int
    eval_every::Int
    lr::Float32
    latent_dim::Int
    kl_weight::Float32
    latent_lr_multiplier::Float32
end

mutable struct TrainingSession{M, Mo}
    model::M
    mode::Mo
    config::TrainingConfig
    tstate::Any
    train_snaps::Array{Float32,3}
    n_train::Int
    lr_sched::Union{Nothing,LRScheduler}
    rng::Any
    sampler::BatchSampler
    compiled_eval::Any
    eval_omega::Array{Float32,3}
    train_losses::Vector{Float32}
    eval_swds::Vector{Float32}
    eval_epochs::Vector{Int}
    kl_weight_ref::Ref{Float32} # current effective KL weight; updated at the top of each epoch
end

function TrainingSession(
    rng,
    model::Lux.AbstractLuxLayer,
    ps, st,
    train_snaps::Array{Float32,3},
    eval_snaps::Array{Float32,3},
    mode;
    latent_dim::Int,
    n_epochs::Int,
    eval_every::Int,
    n_slices::Int,
    lr::Float32,
    lr_scheduler::Symbol             = :reduce_on_plateau,
    plateau_patience::Int            = 10,
    plateau_factor::Float32          = 0.5f0,
    plateau_min_lr::Float32          = 1f-6,
    cosine_T_max::Int                = n_epochs,
    cosine_eta_min::Float32          = 1f-6,
    cosine_T_mult::Int               = 1,
    fix_thetas::Bool                 = true,
    kl_weight::Float32               = 0f0,
    latent_lr_multiplier::Float32    = 10f0,
    init_latent_mu::Union{Nothing,Matrix{Float32},Float32}        = nothing,
    init_latent_log_sigma::Union{Nothing,Matrix{Float32},Float32} = nothing,
)
    n_train          = size(train_snaps, 3)
    latent_mu        = init_latent_mu        isa Float32 ? fill(init_latent_mu, latent_dim, n_train)  :
                       init_latent_mu        !== nothing ? init_latent_mu        : randn(rng, Float32, latent_dim, n_train)
    latent_log_sigma = init_latent_log_sigma isa Float32 ? fill(init_latent_log_sigma, latent_dim, n_train) :
                       init_latent_log_sigma !== nothing ? init_latent_log_sigma : zeros(Float32, latent_dim, n_train)
    ps = merge(ps, (latent_mu        = latent_mu,
                    latent_log_sigma = latent_log_sigma))

    dev = Lux.reactant_device()
    ps  = ps |> dev
    st  = st |> dev

    opt    = build_optimizer(lr)
    tstate = Training.TrainState(model, ps, st, opt)

    # Apply latent LR multiplier to variational parameter subtrees from the start.
    _adjust_latent_lr!(tstate.optimizer_state, lr * latent_lr_multiplier)

    lr_sched = if lr_scheduler == :reduce_on_plateau
        ReduceOnPlateau(lr; factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr)
    elseif lr_scheduler == :cosine_annealing
        CosineAnnealingLR(lr; T_max=cosine_T_max, eta_min=cosine_eta_min)
    elseif lr_scheduler == :cosine_annealing_warm_restarts
        CosineAnnealingWarmRestarts(lr; T_0=cosine_T_max, T_mult=cosine_T_mult, eta_min=cosine_eta_min)
    else
        nothing
    end

    sampler = BatchSampler(rng, mode.batch_size, mode.n_full, n_slices,
                           mode.thetas_flat_dim; fix_thetas=fix_thetas)

    n_eval     = min(size(eval_snaps, 3), mode.batch_size)
    eval_omega = eval_snaps[:, :, 1:n_eval]

    cfg = TrainingConfig(n_epochs, eval_every, lr, latent_dim,
                         kl_weight, latent_lr_multiplier)

    return TrainingSession(
        model, mode, cfg, tstate, train_snaps, n_train, lr_sched, rng,
        sampler, nothing, eval_omega,
        Float32[], Float32[], Int[],
        Ref(0f0),
    )
end

function train!(session::TrainingSession)
    _train!(session, session.mode)
end

# ---------------------------------------------------------------------------
# Phase-1 training loop (VorticityMode and ObservationsMode).
# ---------------------------------------------------------------------------
function _train!(session::TrainingSession, mode::Union{VorticityMode, ObservationsMode})
    T   = Float32
    cfg = session.config

    for epoch in 1:cfg.n_epochs
        session.kl_weight_ref[] = cfg.kl_weight
        kl_w_ra = Reactant.to_rarray(Float32[session.kl_weight_ref[]])

        if !mode.freeze_upsampler
            full_batches = filter(b -> length(b) == mode.batch_size,
                                  DataPipeline.batch_partition(session.n_train, mode.batch_size, session.rng))
            data_all = mode.prepare_epoch(full_batches)
            sample_epoch!(session.sampler)

            batch_losses = Vector{Any}(undef, mode.n_full)
            for i in 1:mode.n_full
                thetas_batch, cols = get_batch(session.sampler, i)
                thetas     = Reactant.to_rarray(thetas_batch)
                cols_ra    = Reactant.to_rarray(collect(Int32, cols))
                eps_ra     = Reactant.to_rarray(randn(session.rng, Float32, cfg.latent_dim, length(cols)))
                data_batch = mode.get_data_batch(data_all, cols)
                _, loss, _, session.tstate = Training.single_train_step!(
                    Lux.AutoEnzyme(), mode.train_loss,
                    (data_batch..., thetas, cols_ra, eps_ra, kl_w_ra), session.tstate)
                batch_losses[i] = loss
            end

            avg_loss = sum(Float32(l) for l in batch_losses) / mode.n_full
            push!(session.train_losses, avg_loss)

            if session.lr_sched !== nothing
                new_lr = step!(session.lr_sched, epoch, avg_loss)
                Optimisers.adjust!(session.tstate.optimizer_state, eta=new_lr)
                _adjust_latent_lr!(session.tstate.optimizer_state, new_lr * cfg.latent_lr_multiplier)
                current_lr = new_lr
            else
                current_lr = cfg.lr
            end
            latent_lr = current_lr * cfg.latent_lr_multiplier
            println("epoch $epoch  loss = $avg_loss  lr = $current_lr  latent_lr = $latent_lr")
        else
            println("epoch $epoch  (upsampler frozen — skipping gradient update)")
        end

        if epoch % cfg.eval_every == 0
            _run_eval!(session, epoch, T)
        end
    end
end


function _run_eval!(session::TrainingSession, epoch::Int, T)
    N_out  = size(session.eval_omega, 1)
    n_eval = size(session.eval_omega, 3)
    x_eval = Reactant.to_rarray(randn(session.rng, T, session.config.latent_dim, n_eval))
    if session.compiled_eval === nothing
        session.compiled_eval = Reactant.@compile eval_decoder_vort(
            session.model, N_out, x_eval,
            session.tstate.parameters, session.tstate.states)
        println("Compiled eval forward pass.")
    end
    gen_omega_ra, _ = session.compiled_eval(
        session.model, N_out, x_eval,
        session.tstate.parameters, session.tstate.states)
    gen_omega   = Array(gen_omega_ra)
    flat_dim    = N_out * N_out
    gen_flat    = reshape(gen_omega,          flat_dim, n_eval)
    eval_flat   = reshape(session.eval_omega, flat_dim, n_eval)
    thetas_eval = randn(session.rng, T, session.sampler.n_slices, flat_dim)
    swd = sliced_wasserstein(gen_flat, eval_flat, thetas_eval)
    push!(session.eval_swds, swd)
    push!(session.eval_epochs, epoch)
    println("  eval SWD = $swd")
end
