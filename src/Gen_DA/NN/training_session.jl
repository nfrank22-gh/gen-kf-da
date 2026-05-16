import Reactant
import Lux
import Lux.Training as Training
import Optimisers

mutable struct TrainingSession{M}
    model::M
    tstate::Any
    train_snaps::Array{Float32,3}
    n_train::Int
    plateau_sched::Union{Nothing,ReduceOnPlateau}
    lr::Float32
    n_epochs::Int
    eval_every::Int
    n_full::Int
    batch_size::Int
    rng::Any
    prepare_epoch::Any    # full_batches -> data_all (mode-specific, device)
    get_data_batch::Any   # (data_all, cols) -> data_tuple for loss
    train_loss::Any       # (model, ps, st, data) -> (loss, st, ())
    sampler::BatchSampler
    compiled_eval::Any
    eval_omega::Array{Float32,3}
    train_losses::Vector{Float32}
    eval_swds::Vector{Float32}
    eval_epochs::Vector{Int}
    freeze_upsampler::Bool
end

function TrainingSession(
    rng,
    model::Lux.AbstractLuxLayer,
    ps, st,
    train_snaps::Array{Float32,3},
    eval_snaps::Array{Float32,3};
    training_mode::Symbol,
    N::Int,
    latent_dim::Int,
    n_epochs::Int,
    eval_every::Int,
    batch_size::Int,
    n_slices::Int,
    lr::Float32,
    use_reduce_on_plateau::Bool = true,
    plateau_patience::Int       = 10,
    plateau_factor::Float32     = 0.5f0,
    plateau_min_lr::Float32     = 1f-6,
    fix_x::Bool                 = true,
    fix_thetas::Bool            = true,
    sensor_ci                   = nothing,
    n_meas_space::Int           = 0,
    rhs_relax                   = nothing,
    n_steps_relax::Int          = 0,
    dt_relax::Float32           = 0.01f0,
    freeze_upsampler::Bool      = false,
)
    dev = Lux.reactant_device()
    ps  = ps |> dev
    st  = st |> dev

    opt    = build_optimizer(lr)
    tstate = Training.TrainState(model, ps, st, opt)

    plateau_sched = use_reduce_on_plateau ?
        ReduceOnPlateau(lr; factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr) :
        nothing

    n_train = size(train_snaps, 3)
    n_full  = n_train ÷ batch_size

    if training_mode == :observations
        data_grid  = SpectralGrid(N)
        sensor_lin = LinearIndices((N, N))[sensor_ci]
        prepare_epoch_fn = function(full_batches)
            u_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
            v_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                cols = (i-1)*batch_size+1 : i*batch_size
                u_cpu[:, cols], v_cpu[:, cols] =
                    DataPipeline.extract_observations(train_snaps, batch_indices, sensor_ci, data_grid)
            end
            return u_cpu, v_cpu   # stay on CPU; uploaded per batch in get_data_batch
        end
        get_data_batch_fn = function(data_all, cols)
            u_all, v_all = data_all
            return (Reactant.to_rarray(u_all[:, cols]), Reactant.to_rarray(v_all[:, cols]))
        end
        train_loss_fn = function(m, params, states, data)
            x, u_trg, v_trg, thetas = data
            l, new_st = loss_fn(m, N, x, params, states, u_trg, v_trg, sensor_lin, thetas)
            l, new_st, (;)
        end
        thetas_flat_dim = 2 * n_meas_space
    elseif training_mode == :vorticity
        prepare_epoch_fn = function(full_batches)
            omega_cpu = zeros(Float32, N, N, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                slabs = (i-1)*batch_size+1 : i*batch_size
                omega_cpu[:, :, slabs] = train_snaps[:, :, batch_indices]
            end
            return (omega_cpu,)   # stay on CPU; uploaded per batch in get_data_batch
        end
        get_data_batch_fn = function(data_all, cols)
            omega_all, = data_all
            return (Reactant.to_rarray(omega_all[:, :, cols]),)
        end
        train_loss_fn = function(m, params, states, data)
            x, omega_trg, thetas = data
            l, new_st = loss_fn_vort_state(m, N, x, params, states, omega_trg, thetas)
            l, new_st, (;)
        end
        thetas_flat_dim = N * N

    else  # :relaxation — decode → KF relax → SWD on relaxed field
        @assert rhs_relax !== nothing "rhs_relax must be provided for training_mode = :relaxation"
        @assert n_steps_relax > 0    "n_steps_relax must be > 0 for training_mode = :relaxation"
        prepare_epoch_fn = function(full_batches)
            omega_cpu = zeros(Float32, N, N, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                slabs = (i-1)*batch_size+1 : i*batch_size
                omega_cpu[:, :, slabs] = train_snaps[:, :, batch_indices]
            end
            return (omega_cpu,)
        end
        get_data_batch_fn = function(data_all, cols)
            omega_all, = data_all
            return (Reactant.to_rarray(omega_all[:, :, cols]),)
        end
        train_loss_fn = function(m, params, states, data)
            x, omega_trg, thetas = data
            l, new_st = loss_fn_relaxation(m, N, x, params, states,
                                           rhs_relax, n_steps_relax, dt_relax,
                                           omega_trg, thetas)
            l, new_st, (;)
        end
        thetas_flat_dim = N * N
    end

    sampler = BatchSampler(rng, latent_dim, batch_size, n_full, n_slices, thetas_flat_dim;
                           fix_x=fix_x, fix_thetas=fix_thetas)

    n_eval     = min(size(eval_snaps, 3), batch_size)
    eval_omega = eval_snaps[:, :, 1:n_eval]

    # compiled_eval is initialised lazily in train!() on the first eval call,
    # after single_train_step! has already locked in its gradient compilation.
    # Pre-compiling here interferes with the gradient thunk for custom model types.
    return TrainingSession(
        model, tstate, train_snaps, n_train, plateau_sched, lr,
        n_epochs, eval_every, n_full, batch_size, rng,
        prepare_epoch_fn, get_data_batch_fn, train_loss_fn,
        sampler, nothing, eval_omega,
        Float32[], Float32[], Int[],
        freeze_upsampler,
    )
end

function train!(session::TrainingSession)
    T = Float32
    for epoch in 1:session.n_epochs
        if !session.freeze_upsampler
            full_batches = filter(b -> length(b) == session.batch_size,
                                  DataPipeline.batch_partition(session.n_train, session.batch_size, session.rng))

            data_all = session.prepare_epoch(full_batches)

            x_cpu, thetas_cpu = sample_epoch!(session.sampler)
            # Both x and thetas kept on CPU; each batch slice is uploaded just-in-time
            # so that the Enzyme backward pass (which holds activations) doesn't compete
            # with extra full-epoch GPU buffers for memory.

            batch_losses = Vector{Any}(undef, session.n_full)
            for i in 1:session.n_full
                x_batch, thetas_batch, cols = get_batch(session.sampler, x_cpu, thetas_cpu, i)
                x      = Reactant.to_rarray(x_batch)
                thetas = Reactant.to_rarray(thetas_batch)
                data_batch = session.get_data_batch(data_all, cols)
                _, loss, _, session.tstate = Training.single_train_step!(
                    Lux.AutoEnzyme(), session.train_loss, (x, data_batch..., thetas), session.tstate)
                batch_losses[i] = loss
            end

            avg_loss   = sum(Float32(l) for l in batch_losses) / session.n_full
            current_lr = session.plateau_sched !== nothing ? session.plateau_sched.current_lr : session.lr
            push!(session.train_losses, avg_loss)
            println("epoch $epoch  loss = $avg_loss  lr = $current_lr")

            if session.plateau_sched !== nothing
                new_lr = step!(session.plateau_sched, avg_loss)
                Optimisers.adjust!(session.tstate.optimizer_state, eta=new_lr)
            end
        else
            println("epoch $epoch  (upsampler frozen — skipping gradient update)")
        end

        if epoch % session.eval_every == 0
            N_out  = size(session.eval_omega, 1)
            n_eval = size(session.eval_omega, 3)
            x_eval = Reactant.to_rarray(randn(session.rng, T, session.sampler.latent_dim, n_eval))
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
    end
end
