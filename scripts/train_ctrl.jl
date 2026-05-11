using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Reactant, Random, Lux, Lux.Training, Optimisers, FFTW, Enzyme

Reactant.set_default_backend("cuda")
const dev = reactant_device()

function main()
    T             = Float32
    Re            = 100
    N             = 128
    npart         = 40
    T_train       = 800.0f0
    n_meas_space  = 20
    batch_size    = 100
    n_epochs      = 200
    eval_every    = 10
    num_freq      = 16
    latent_dim    = 100
    layers        = [latent_dim, 512, 1024]
    n_slices      = 50
    lr            = 1f-3
    use_cosine_lr = true

    rng = Xoshiro(123)

    # ── Data ──────────────────────────────────────────────────────────────────
    traj_data = load_trajectory(
        "data/particles/Re$(Re)_N$(N)_npart$(npart)/trajectory.jld2")
    train_snaps, eval_snaps = split_trajectory(
        traj_data.trajectory, traj_data.dt, traj_data.save_every, T_train)
    n_train = size(train_snaps, 3)
    println("Training on $n_train snapshots, eval on $(size(eval_snaps, 3)) snapshots")

    # ── Sensor array (fixed throughout training) ───────────────────────────────
    sensor_ci  = make_sensor_array(N, n_meas_space, rng)
    sensor_lin = LinearIndices((N, N))[sensor_ci]  # kept as CPU Int array (Reactant constant)

    # ── Model and optimizer ────────────────────────────────────────────────────
    model, ps, st = VortFourierDecoder(layers, num_freq, T(2π), rng, T)
    ps = ps |> dev
    st = st |> dev

    opt, lr_schedule = build_optimizer(lr, n_epochs, use_cosine_lr)
    tstate = Training.TrainState(model, ps, st, opt)

    # Loss closure: model, N, sensor_lin are compile-time constants for XLA
    function train_loss(m, params, states, data)
        x, u_trg, v_trg, thetas = data
        l, new_st = loss_fn(m, N, x, params, states, u_trg, v_trg, sensor_lin, thetas)
        return l, new_st, (;)
    end

    # ── Compilation ────────────────────────────────────────────────────────────
    x_dummy      = Reactant.to_rarray(zeros(T, latent_dim, batch_size))
    u_dummy      = Reactant.to_rarray(zeros(T, n_meas_space, batch_size))
    v_dummy      = Reactant.to_rarray(zeros(T, n_meas_space, batch_size))
    thetas_dummy = Reactant.to_rarray(zeros(T, n_slices, 2 * n_meas_space))
    data_dummy   = (x_dummy, u_dummy, v_dummy, thetas_dummy)

    # Compile the full training step: forward + Enzyme backward + Adam update
    function step_fn(data, ts)
        return Training.single_train_step!(AutoEnzyme(), train_loss, data, ts)
    end
    compiled_step = @compile step_fn(data_dummy, tstate)
    println("Compiled training step.")

    # Compile eval forward pass (decoder to omega_hat at model resolution)
    x_eval_dummy  = Reactant.to_rarray(zeros(T, latent_dim, batch_size))
    compiled_eval = @compile eval_decoder_vort_hat(
        model, x_eval_dummy, tstate.parameters, tstate.states)
    println("Compiled eval forward pass.")

    # ── Precompute truncated spectral eval targets ─────────────────────────────
    # Map full-N rfft to model's native resolution for SWD comparison
    NDOF_model  = 2 * num_freq - 1      # 31
    nfreq_model = num_freq              # 16
    half_model  = NDOF_model ÷ 2       # 15  (matches spectral_pad's half_in)
    n_eval      = min(size(eval_snaps, 3), batch_size)
    eval_omega_hat = zeros(Complex{T}, nfreq_model, NDOF_model, n_eval)
    for i in 1:n_eval
        oh = rfft(@view eval_snaps[:, :, i])          # (N÷2+1, N)
        eval_omega_hat[:, 1:half_model, i]            .= oh[1:nfreq_model, 1:half_model]
        eval_omega_hat[:, half_model+2:NDOF_model, i] .= oh[1:nfreq_model, N-half_model+1:N]
        # col half_model+1 (x-freq 15, Nyquist for NDOF_model) stays zero — matches spectral_pad
    end

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses = T[]
    eval_swds    = T[]
    eval_epochs  = Int[]

    for epoch in 1:n_epochs
        # Apply cosine LR schedule before each epoch
        if lr_schedule !== nothing
            Optimisers.adjust!(tstate.optimizer_state, eta=lr_schedule(epoch))
        end

        batches    = batch_partition(n_train, batch_size, rng)
        epoch_loss = zero(T)
        n_full     = 0

        for batch_indices in batches
            length(batch_indices) == batch_size || continue

            u_trg, v_trg = extract_observations(train_snaps, batch_indices, sensor_ci, N)
            # Fresh latents and projection directions sampled CPU-side, moved to device
            x      = Reactant.to_rarray(randn(rng, T, latent_dim, batch_size))
            thetas = Reactant.to_rarray(randn(rng, T, n_slices, 2 * n_meas_space))
            u_dev  = Reactant.to_rarray(u_trg)
            v_dev  = Reactant.to_rarray(v_trg)

            _, loss, _, tstate = compiled_step((x, u_dev, v_dev, thetas), tstate)

            epoch_loss += Array(loss)[]
            n_full += 1
        end

        avg_loss = epoch_loss / max(n_full, 1)
        push!(train_losses, avg_loss)
        println("epoch $epoch  loss = $avg_loss")

        # ── Eval ──────────────────────────────────────────────────────────────
        if epoch % eval_every == 0
            x_eval = Reactant.to_rarray(randn(rng, T, latent_dim, n_eval))
            oh_re, oh_im, _ = compiled_eval(
                model, x_eval, tstate.parameters, tstate.states)
            gen_omega_hat = complex.(Array(oh_re), Array(oh_im))
            swd = sliced_wasserstein_spectral(gen_omega_hat, eval_omega_hat, n_slices; rng)
            push!(eval_swds, swd)
            push!(eval_epochs, epoch)
            println("  eval SWD = $swd")
        end
    end

    # ── Checkpoint ────────────────────────────────────────────────────────────
    config = Dict{String, Any}(
        "Re" => Re, "N" => N, "npart" => npart,
        "T_train" => T_train, "n_meas_space" => n_meas_space,
        "batch_size" => batch_size, "n_epochs" => n_epochs,
        "eval_every" => eval_every, "num_freq" => num_freq,
        "latent_dim" => latent_dim, "layers" => layers,
        "n_slices" => n_slices, "lr" => lr, "use_cosine_lr" => use_cosine_lr,
        "sensor_locations" => sensor_ci,
    )
    save_checkpoint(
        "checkpoints/Re$(Re)_N$(N)",
        tstate.parameters, tstate.states,
        train_losses, eval_swds, eval_epochs,
        config)
    println("Training complete. Checkpoint saved.")
end

main()
