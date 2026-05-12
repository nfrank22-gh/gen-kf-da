using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Gen_DA.NN.TrainingPlots
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
    batch_size    = 400
    n_epochs      = 1000
    eval_every    = 10
    num_freq      = 8
    latent_dim    = 200
    layers        = [latent_dim, 256, 512]
    n_slices      = 10000
    lr                    = 1f-1
    use_reduce_on_plateau = true
    plateau_patience     = 10
    plateau_factor       = 0.5f0
    plateau_min_lr       = 1f-6
    fix_thetas    = true
    fix_x         = true
    # :observations — sparse velocity at sensor locations (production)
    # :vorticity    — full spectral vorticity fields (testing simplification)
    training_mode = :vorticity

    rng = Xoshiro(123)

    # ── Data ──────────────────────────────────────────────────────────────────
    traj_path = "data/no_particles/Re=$(Re)_N=$(N)_dt=$(data_dt)_T=$(T_data)/trajectory.jld2"
    model_dir = joinpath(dirname(traj_path), "model")
    traj_data = load_trajectory(traj_path)

    train_snaps, eval_snaps = split_trajectory(
        traj_data.trajectory, traj_data.dt, traj_data.save_every, T_train)
    println("Training on $(size(train_snaps, 3)) snapshots, eval on $(size(eval_snaps, 3)) snapshots")

    # ── Sensor array (observations mode only) ─────────────────────────────────
    sensor_ci = training_mode == :observations ? make_sensor_array(N, n_meas_space, rng) : nothing

    # ── Model ─────────────────────────────────────────────────────────────────
    model, ps, st = VortFourierDecoder(layers, num_freq, rng, T)

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
        "T_train" => T_train, "n_meas_space" => n_meas_space,
        "batch_size" => batch_size, "n_epochs" => n_epochs,
        "eval_every" => eval_every, "num_freq" => num_freq,
        "latent_dim" => latent_dim, "layers" => layers,
        "n_slices" => n_slices, "lr" => lr,
        "use_reduce_on_plateau" => use_reduce_on_plateau,
        "plateau_patience" => plateau_patience, "plateau_factor" => plateau_factor,
        "plateau_min_lr" => plateau_min_lr,
        "fix_thetas" => fix_thetas,
        "fix_x" => fix_x,
        "training_mode" => string(training_mode),
    )
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
    NDOF_model  = 2 * num_freq - 1
    nfreq_model = num_freq
    n_plot = 4
    x_plot = randn(rng, T, latent_dim, n_plot)
    gen_omega, _ = eval_decoder_vort(model, N, x_plot, ps_cpu, st_cpu)
    gt_idx   = rand(rng, 1:size(eval_snaps, 3), n_plot)
    gt_omega = eval_snaps[:, :, gt_idx]
    plot_vorticity_panel(gen_omega, gt_omega, model_dir)

    gt_oh_re, gt_oh_im = extract_vorticity_spectral(
        eval_snaps, collect(1:n_plot), nfreq_model, NDOF_model, N)
    gt_omega_hat  = complex.(gt_oh_re, gt_oh_im)
    gen_oh_re, gen_oh_im, _ = NN._decode_vort_hat(model, x_plot, ps_cpu, st_cpu)
    gen_omega_hat = complex.(gen_oh_re, gen_oh_im)
    plot_energy_spectrum(gen_omega_hat, gt_omega_hat, model_dir)

    println("Training complete. Checkpoint and plots saved to $model_dir")
end

main()
