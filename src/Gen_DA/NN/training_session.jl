import Reactant
import Lux
import Lux.Training as Training
import Optimisers

mutable struct TrainingSession{M, Mo}
    model::M
    mode::Mo
    tstate::Any
    train_snaps::Array{Float32,3}
    n_train::Int
    plateau_sched::Union{Nothing,ReduceOnPlateau}
    lr::Float32
    n_epochs::Int
    eval_every::Int
    rng::Any
    sampler::BatchSampler
    compiled_eval::Any
    eval_omega::Array{Float32,3}
    train_losses::Vector{Float32}
    eval_swds::Vector{Float32}
    eval_epochs::Vector{Int}
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
    use_reduce_on_plateau::Bool = true,
    plateau_patience::Int       = 10,
    plateau_factor::Float32     = 0.5f0,
    plateau_min_lr::Float32     = 1f-6,
    fix_x::Bool                 = true,
    fix_thetas::Bool            = true,
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

    sampler = BatchSampler(rng, latent_dim, mode.batch_size, mode.n_full, n_slices,
                           mode.thetas_flat_dim; fix_x=fix_x, fix_thetas=fix_thetas)

    n_eval     = min(size(eval_snaps, 3), mode.batch_size)
    eval_omega = eval_snaps[:, :, 1:n_eval]

    # compiled_eval is initialised lazily on the first eval call, after
    # single_train_step! has locked in its gradient compilation.
    return TrainingSession(
        model, mode, tstate, train_snaps, n_train, plateau_sched, lr,
        n_epochs, eval_every, rng,
        sampler, nothing, eval_omega,
        Float32[], Float32[], Int[],
    )
end

function train!(session::TrainingSession)
    _train!(session, session.mode)
end

# ---------------------------------------------------------------------------
# Phase-1 training loop (VorticityMode and ObservationsMode).
# ---------------------------------------------------------------------------
function _train!(session::TrainingSession, mode::Union{VorticityMode, ObservationsMode})
    T = Float32
    for epoch in 1:session.n_epochs
        if !mode.freeze_upsampler
            full_batches = filter(b -> length(b) == mode.batch_size,
                                  DataPipeline.batch_partition(session.n_train, mode.batch_size, session.rng))
            data_all = mode.prepare_epoch(full_batches)
            sample_epoch!(session.sampler)

            batch_losses = Vector{Any}(undef, mode.n_full)
            for i in 1:mode.n_full
                x_batch, thetas_batch, cols = get_batch(session.sampler, i)
                x      = Reactant.to_rarray(x_batch)
                thetas = Reactant.to_rarray(thetas_batch)
                data_batch = mode.get_data_batch(data_all, cols)
                _, loss, _, session.tstate = Training.single_train_step!(
                    Lux.AutoEnzyme(), mode.train_loss, (x, data_batch..., thetas), session.tstate)
                batch_losses[i] = loss
            end

            avg_loss   = sum(Float32(l) for l in batch_losses) / mode.n_full
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
            _run_eval!(session, epoch, T)
        end
    end
end

# ---------------------------------------------------------------------------
# Phase-2 training loop (RelaxationMode).
# Bypasses Training.single_train_step! for the solver chain; uses it only
# for the upsampler VJP (which has no while loops and compiles fine).
#
# Per-epoch flow:
#   1. Forward (GPU, compiled): upsampler + relax_and_store  →  traj, omega_relaxed
#   2. SWD + adjoint (CPU, analytical): sliced_wasserstein + sliced_wasserstein_adjoint
#   3. Backward (GPU, compiled): relax_adj_full               →  d_omega_gen
#   4. VJP update (GPU, Enzyme on upsampler only): update tstate via inner-product trick
# ---------------------------------------------------------------------------
function _train!(session::TrainingSession, mode::RelaxationMode)
    T          = Float32
    rhs        = mode.rhs_relax
    n_steps    = mode.n_steps_relax
    dt         = mode.dt_relax
    N_out      = mode.N_out
    flat_dim   = N_out * N_out
    latent_dim = session.sampler.latent_dim
    batch_size = mode.batch_size

    # VJP loss: gradient of L = ⟨omega_gen, d_omega_gen⟩ w.r.t. params equals
    # J_upsampler^T * d_omega_gen — the exact VJP we need to drive the solver adjoint.
    vjp_loss = (m, params, states, data) -> begin
        x_, d_ω = data
        ω, new_st = eval_decoder_vort(m, N_out, x_, params, states)
        l = sum(ω .* d_ω)
        return l, new_st, (;)
    end

    # Pre-compile forward and adjoint kernels using dummy arrays of the correct shape.
    # n_steps is a compile-time constant; rhs values are embedded as XLA constants.
    println("Compiling phase-2 forward (relax_and_store, $n_steps steps)...")
    let x_dummy = Reactant.to_rarray(zeros(T, latent_dim, batch_size)),
        ps      = session.tstate.parameters,
        st      = session.tstate.states
        mode.compiled_relax_fwd = Reactant.@compile _phase2_forward(
            session.model, ps, st, x_dummy, rhs, n_steps, dt, N_out)
    end
    println("Compiling phase-2 adjoint (relax_adj_full, $n_steps steps)...")
    let traj_dummy = Reactant.to_rarray(zeros(ComplexF32, N_out÷2+1, N_out, batch_size, n_steps+1)),
        d_ω_dummy  = Reactant.to_rarray(zeros(T, N_out, N_out, batch_size))
        mode.compiled_relax_adj = Reactant.@compile relax_adj_full(
            rhs, traj_dummy, n_steps, dt, d_ω_dummy, N_out)
    end

    for epoch in 1:session.n_epochs
        full_batches = filter(b -> length(b) == mode.batch_size,
                              DataPipeline.batch_partition(session.n_train, mode.batch_size, session.rng))
        data_all = mode.prepare_epoch(full_batches)
        sample_epoch!(session.sampler)

        batch_losses = Vector{Float32}(undef, mode.n_full)

        for i in 1:mode.n_full
            x_batch, thetas_batch, cols = get_batch(session.sampler, i)
            x_ra = Reactant.to_rarray(x_batch)

            # 1. Forward on GPU ─────────────────────────────────────────────
            traj_ra, ω_relaxed_ra = mode.compiled_relax_fwd(
                session.model, session.tstate.parameters, session.tstate.states,
                x_ra, rhs, n_steps, dt, N_out)

            # 2. SWD + gradient on CPU ──────────────────────────────────────
            omega_data_batch = mode.get_data_batch(data_all, cols)
            omega_trg_ra     = omega_data_batch[1]

            ω_relaxed_cpu = Array(ω_relaxed_ra)
            ω_trg_cpu     = Array(omega_trg_ra)

            P_cpu = reshape(ω_relaxed_cpu, flat_dim, size(ω_relaxed_cpu, 3))
            Q_cpu = reshape(ω_trg_cpu,     flat_dim, size(ω_trg_cpu,     3))

            swd_val         = Float32(sliced_wasserstein(P_cpu, Q_cpu, thetas_batch))
            d_P_cpu         = sliced_wasserstein_adjoint(P_cpu, Q_cpu, thetas_batch)
            d_ω_relaxed_cpu = reshape(d_P_cpu, N_out, N_out, size(P_cpu, 2))

            # 3. Solver adjoint on GPU ──────────────────────────────────────
            d_ω_relaxed_ra = Reactant.to_rarray(d_ω_relaxed_cpu)
            d_ω_gen_ra = mode.compiled_relax_adj(
                rhs, traj_ra, n_steps, dt, d_ω_relaxed_ra, N_out)

            # 4. Upsampler VJP update ───────────────────────────────────────
            # Training.single_train_step! computes d_params = J^T * d_ω_gen
            # and applies the Adam step.  The "loss" returned is the inner
            # product (not the SWD) — we record swd_val instead.
            _, _, _, session.tstate = Training.single_train_step!(
                Lux.AutoEnzyme(), vjp_loss, (x_ra, d_ω_gen_ra), session.tstate)

            batch_losses[i] = swd_val
        end

        avg_swd    = mean(batch_losses)
        current_lr = session.plateau_sched !== nothing ? session.plateau_sched.current_lr : session.lr
        push!(session.train_losses, avg_swd)
        println("epoch $epoch  swd = $avg_swd  lr = $current_lr")

        if session.plateau_sched !== nothing
            new_lr = step!(session.plateau_sched, avg_swd)
            Optimisers.adjust!(session.tstate.optimizer_state, eta=new_lr)
        end

        if epoch % session.eval_every == 0
            _run_eval!(session, epoch, T)
        end
    end
end

# Forward wrapper compiled in _train! before the epoch loop.
# Returns (traj, omega_relaxed) where traj has shape (N÷2+1, N, batch, n_steps+1).
function _phase2_forward(model, ps, st, x, rhs, n_steps, dt, N_out)
    omega_gen, _ = eval_decoder_vort(model, N_out, x, ps, st)
    omega_hat_0  = rfft(omega_gen, 1:2)
    traj         = relax_and_store(rhs, omega_hat_0, n_steps, dt)
    omega_relaxed = irfft(traj[:, :, :, n_steps + 1], N_out, 1:2)
    return traj, omega_relaxed
end

function _run_eval!(session::TrainingSession, epoch::Int, T)
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
