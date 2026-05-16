using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Gen_DA.NN.TrainingPlots
using Gen_DA.Solver: KfRhs
using Reactant, Random, Lux

Reactant.set_default_backend("cuda")

function main()
    T             = Float32
    Re            = 100
    N             = 128
    data_dt       = 0.01
    T_data        = 10000
    T_train       = 8000.0f0
    n_meas_space  = 50          # only used when training_mode == :observations
    batch_size    = 801
    n_epochs      = 10
    eval_every    = 100
    latent_dim    = 16
    n_slices      = 10000
    lr                    = 1f-3
    use_reduce_on_plateau = true
    plateau_patience     = 10
    plateau_factor       = 0.5f0
    plateau_min_lr       = 1f-6
    fix_thetas    = true
    fix_x         = false
    # :observations — sparse velocity at sensor locations (production)
    # :vorticity    — full spectral vorticity fields (testing simplification)
    training_mode = :vorticity

    # ── Phase 2: Relaxation training ──────────────────────────────────────────
    # Runs the KF solver for T_relax from the upsampler output, then computes
    # SWD on the relaxed field.  Backprop flows through the short solver rollout
    # into the upsampler weights.  Set run_phase2 = false to skip entirely.
    run_phase2        = true
    T_relax           = 2f0   # physical time to run the solver per sample
    dt_relax          = 0.01f0  # solver timestep for relaxation
    n_epochs_phase2   = 2
    eval_every_phase2 = 10
    lr_phase2         = 1f-3
    freeze_upsampler  = false   # no-op when fine-tuner has no learnable params

    # ── Model ─────────────────────────────────────────────────────────────────
    # :fourier — VortFourierDecoder (MLP → spectral coefficients → IFFT)
    # :conv    — ConvDecoder (FC → spatial feature map → spectral upsampling blocks)
    model_arch = :conv

    rng = Xoshiro(123)

    if model_arch == :fourier
        num_freq = 8                          # spectral resolution; NDOF = 2·num_freq − 1
        layers   = [latent_dim, 256, 512]     # MLP hidden widths (first entry must equal latent_dim)
        model, ps, st = VortFourierDecoder(layers, num_freq, rng, T)

    elseif model_arch == :conv
        act               = gelu
        init_channels     = 32
        fc_hidden         = [128]             # intermediate FC widths; final output is derived
        n_upsample_blocks = 3                 # starting resolution = N ÷ 2^n_upsample_blocks = 8
        conv_channels     = [32, 16, 8]  # C_out after 1×1 proj in each upsampling block
        n_convs_per_block = 4
        kernel_size       = 3
        norm_type         = :batch             # :none | :batch | :group
        n_groups          = 8                 # only used when norm_type == :group
        model, ps, st = ConvDecoder(latent_dim, fc_hidden, init_channels,
                                    n_upsample_blocks, conv_channels,
                                    n_convs_per_block, kernel_size,
                                    act, norm_type, n_groups, N, rng, T)
    end

    # ── Data ──────────────────────────────────────────────────────────────────
    traj_path = "data/no_particles/Re=$(Re)_N=$(N)_dt=$(data_dt)_T=$(T_data)/trajectory.jld2"
    model_dir = joinpath(dirname(traj_path), "model")
    traj_data = load_trajectory(traj_path)
    n_forcing = traj_data.n   # Kolmogorov forcing wavenumber from data generation

    train_snaps, eval_snaps = split_trajectory(
        traj_data.trajectory, traj_data.dt, traj_data.save_every, T_train)
    println("Training on $(size(train_snaps, 3)) snapshots, eval on $(size(eval_snaps, 3)) snapshots")

    # ── Sensor array (observations mode only) ─────────────────────────────────
    sensor_ci = training_mode == :observations ? make_sensor_array(N, n_meas_space, rng) : nothing

    # ── Session ───────────────────────────────────────────────────────────────
    session = TrainingSession(
        rng, model, ps, st, train_snaps, eval_snaps;
        training_mode=training_mode, N=N, latent_dim=latent_dim,
        n_meas_space=n_meas_space, sensor_ci=sensor_ci,
        n_epochs=n_epochs, eval_every=eval_every,
        batch_size=batch_size, n_slices=n_slices, lr=lr,
        use_reduce_on_plateau=use_reduce_on_plateau,
        plateau_patience=plateau_patience, plateau_factor=plateau_factor,
        plateau_min_lr=plateau_min_lr,
        fix_x=fix_x, fix_thetas=fix_thetas,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    train!(session)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    config = Dict{String, Any}(
        "Re" => Re, "N" => N,
        "model_arch" => string(model_arch),
        "T_train" => T_train, "n_meas_space" => n_meas_space,
        "batch_size" => batch_size, "n_epochs" => n_epochs,
        "eval_every" => eval_every,
        "latent_dim" => latent_dim,
        "n_slices" => n_slices, "lr" => lr,
        "use_reduce_on_plateau" => use_reduce_on_plateau,
        "plateau_patience" => plateau_patience, "plateau_factor" => plateau_factor,
        "plateau_min_lr" => plateau_min_lr,
        "fix_thetas" => fix_thetas,
        "fix_x" => fix_x,
        "training_mode" => string(training_mode),
    )
    if model_arch == :fourier
        config["num_freq"] = num_freq
        config["layers"]   = layers
    elseif model_arch == :conv
        config["init_channels"]     = init_channels
        config["fc_hidden"]         = fc_hidden
        config["n_upsample_blocks"] = n_upsample_blocks
        config["conv_channels"]     = conv_channels
        config["n_convs_per_block"] = n_convs_per_block
        config["kernel_size"]       = kernel_size
        config["norm_type"]         = string(norm_type)
        config["n_groups"]          = n_groups
    end
    if training_mode == :observations
        config["sensor_locations"] = sensor_ci
    end
    save_checkpoint(model_dir,
        session.tstate.parameters, session.tstate.states,
        session.train_losses, session.eval_swds, session.eval_epochs,
        config)
    println("Checkpoint saved.")

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    plot_train_loss_curve(session.train_losses, model_dir)
    plot_eval_swd_curve(session.eval_swds, session.eval_epochs, model_dir)

    cpu    = Lux.cpu_device()
    ps_cpu = cpu(session.tstate.parameters)
    st_cpu = cpu(session.tstate.states)
    n_plot = 4
    x_plot = randn(rng, T, latent_dim, n_plot)
    gen_omega, _ = eval_decoder_vort(model, N, x_plot, ps_cpu, st_cpu)
    gt_idx   = rand(rng, 1:size(eval_snaps, 3), n_plot)
    gt_omega = eval_snaps[:, :, gt_idx]
    plot_vorticity_panel(gen_omega, gt_omega, model_dir)

    if model_arch == :fourier
        NDOF_model  = 2 * num_freq - 1
        nfreq_model = num_freq
        gt_oh_re, gt_oh_im = extract_vorticity_spectral(
            eval_snaps, collect(1:n_plot), nfreq_model, NDOF_model, N)
        gt_omega_hat  = complex.(gt_oh_re, gt_oh_im)
        gen_oh_re, gen_oh_im, _ = NN._decode_vort_hat(model, x_plot, ps_cpu, st_cpu)
        gen_omega_hat = complex.(gen_oh_re, gen_oh_im)
        plot_energy_spectrum(gen_omega_hat, gt_omega_hat, model_dir)
    end

    println("Training complete. Checkpoint and plots saved to $model_dir")

    # ── Phase 2: Relaxation training ──────────────────────────────────────────
    if run_phase2
        n_steps_relax = round(Int, T_relax / dt_relax)
        println("Phase 2: relaxation training ($n_steps_relax solver steps per sample, T_relax=$T_relax)")

        rhs_relax = KfRhs(Re, n_forcing, N)

        # Start phase 2 from the trained upsampler weights (moved to CPU first
        # so the session constructor can re-upload them to the Reactant device).
        cpu    = Lux.cpu_device()
        ps_p2  = cpu(session.tstate.parameters)
        st_p2  = cpu(session.tstate.states)

        # Drop the phase-1 session so its GPU buffers (params + Adam moments +
        # compiled eval) become eligible for collection before phase-2 allocates
        # its own tstate and the per-batch thetas tensor (~1.2 GiB).
        session = nothing
        GC.gc(true)
        GC.gc(true)   # second pass ensures PJRT buffer finalizers run

        session2 = TrainingSession(
            rng, model, ps_p2, st_p2, train_snaps, eval_snaps;
            training_mode=:relaxation, N=N, latent_dim=latent_dim,
            n_epochs=n_epochs_phase2, eval_every=eval_every_phase2,
            batch_size=batch_size, n_slices=n_slices, lr=lr_phase2,
            use_reduce_on_plateau=use_reduce_on_plateau,
            plateau_patience=plateau_patience, plateau_factor=plateau_factor,
            plateau_min_lr=plateau_min_lr,
            fix_x=fix_x, fix_thetas=fix_thetas,
            rhs_relax=rhs_relax, n_steps_relax=n_steps_relax, dt_relax=dt_relax,
            freeze_upsampler=freeze_upsampler,
        )

        train!(session2)

        phase2_dir = joinpath(model_dir, "phase2")
        config_p2 = Dict{String, Any}(
            "Re" => Re, "N" => N,
            "model_arch" => string(model_arch),
            "T_train" => T_train,
            "batch_size" => batch_size, "n_epochs" => n_epochs_phase2,
            "eval_every" => eval_every_phase2,
            "latent_dim" => latent_dim,
            "n_slices" => n_slices, "lr" => lr_phase2,
            "use_reduce_on_plateau" => use_reduce_on_plateau,
            "plateau_patience" => plateau_patience, "plateau_factor" => plateau_factor,
            "plateau_min_lr" => plateau_min_lr,
            "fix_thetas" => fix_thetas, "fix_x" => fix_x,
            "training_mode" => "relaxation",
            "T_relax" => T_relax, "dt_relax" => dt_relax,
            "n_steps_relax" => n_steps_relax,
            "freeze_upsampler" => freeze_upsampler,
        )
        save_checkpoint(phase2_dir,
            session2.tstate.parameters, session2.tstate.states,
            session2.train_losses, session2.eval_swds, session2.eval_epochs,
            config_p2)
        println("Phase 2 checkpoint saved to $phase2_dir")

        !isempty(session2.train_losses) && plot_train_loss_curve(session2.train_losses, phase2_dir)
        !isempty(session2.eval_swds)    && plot_eval_swd_curve(session2.eval_swds, session2.eval_epochs, phase2_dir)

        ps_p2_cpu = cpu(session2.tstate.parameters)
        st_p2_cpu = cpu(session2.tstate.states)
        gen_omega_p2, _ = eval_decoder_vort(model, N, x_plot, ps_p2_cpu, st_p2_cpu)
        plot_vorticity_panel(gen_omega_p2, gt_omega, phase2_dir)

        println("Phase 2 training complete. Results saved to $phase2_dir")
    end
end

main()
