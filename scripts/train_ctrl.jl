using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Gen_DA.NN.TrainingPlots
using Reactant, Random, Lux
import Optimisers

Reactant.set_default_backend("cuda")

function main()
    T             = Float32
    Re            = 40
    N             = 64
    data_dt       = 0.01
    T_data        = 10000
    T_train       = 8000.0f0
    n_meas_space  = 200          # only used when training_mode == :observations
    batch_size    = 801
    n_epochs      = 1000
    eval_every    = 100
    latent_dim    = 40
    n_slices      = 1000
    lr                    = 1f-2
    use_reduce_on_plateau = true
    plateau_patience     = 10
    plateau_factor       = 0.5f0
    plateau_min_lr       = 1f-6
    fix_thetas         = true
    kl_weight          = 1f0   # beta-VAE weight on KL(N(mu,sigma²) || N(0,I))
    # :observations — sparse velocity at sensor locations (production)
    # :vorticity    — full spectral vorticity fields (testing simplification)
    training_mode = :vorticity

    # ── Phase 2: Neural Operator finetuning ───────────────────────────────────
    # Wraps the phase-1 upsampler with a FourierNeuralOperator trained jointly
    # via end-to-end SWD.  The FNO is applied autoregressively n_fno_steps times.
    # Uses the same training_mode as phase 1 (:vorticity or :observations).
    # Set run_phase2 = false to skip entirely.
    #
    # decoder_checkpoint_dir: if set to a directory path, load decoder weights
    # from that checkpoint and skip phase-1 training entirely.  The model
    # architecture must match the hyperparameters below.  Set to nothing to
    # train phase 1 from scratch.
    run_phase2              = false
    decoder_checkpoint_dir  = "data/no_particles/Re=$(Re)_N=$(N)_dt=$(data_dt)_T=$(T_data)/model"   # e.g. "data/no_particles/.../model"
    decoder_checkpoint_dir = nothing
    n_fno_steps             = 4     # autoregressive FNO applications per sample
    n_modes                 = 12     # spectral truncation (modes per x/y direction)
    fno_channels            = 8    # FNO hidden channel width
    n_fno_layers            = 2     # number of FNO layers
    n_epochs_phase2         = 200
    eval_every_phase2       = 100
    lr_phase2               = 1f-3
    freeze_decoder_phase2   = true  # if true, only FNO weights are updated in phase 2
    fix_latents_phase2      = false  # if true, latent matrix is frozen in phase 2
    # Smaller batch for phase 2: FNO intermediates scale with batch×channels×N².
    # Each rfft output is (N÷2+1, N, fno_channels, B) complex; stored per layer per step.
    batch_size_phase2 = 801

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
        init_channels     = 64
        fc_hidden         = [512]             # intermediate FC widths; final output is derived
        n_upsample_blocks = 4                 # starting resolution = N ÷ 2^n_upsample_blocks = 8
        conv_channels     = [64, 32, 16, 8]  # C_out after 1×1 proj in each upsampling block
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

    # ── Eval plot data (shared by phase-1 and phase-2 vorticity panels) ───────
    n_plot   = 4
    # true  → pick n_plot random columns from the trained latent matrix after each phase
    # false → draw fresh samples from N(0,I)
    vort_panel_use_optimized_latents = false
    gt_idx   = rand(rng, 1:size(eval_snaps, 3), n_plot)
    gt_omega = eval_snaps[:, :, gt_idx]

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if decoder_checkpoint_dir === nothing
        mode = if training_mode == :observations
            ObservationsMode(train_snaps, N, batch_size, n_meas_space, sensor_ci; kl_weight=kl_weight)
        else
            VorticityMode(train_snaps, N, batch_size; kl_weight=kl_weight)
        end
        session = TrainingSession(
            rng, model, ps, st, train_snaps, eval_snaps, mode;
            latent_dim=latent_dim, n_epochs=n_epochs, eval_every=eval_every,
            n_slices=n_slices, lr=lr,
            use_reduce_on_plateau=use_reduce_on_plateau,
            plateau_patience=plateau_patience, plateau_factor=plateau_factor,
            plateau_min_lr=plateau_min_lr,
            fix_thetas=fix_thetas, kl_weight=kl_weight,
        )

        train!(session)

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
            "kl_weight" => kl_weight,
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

        plot_train_loss_curve(session.train_losses, model_dir)
        plot_eval_swd_curve(session.eval_swds, session.eval_epochs, model_dir)

        cpu    = Lux.cpu_device()
        ps_p1  = cpu(session.tstate.parameters)
        st_p1  = cpu(session.tstate.states)
        init_latent_mu_p2        = Array(ps_p1.latent_mu)
        init_latent_log_sigma_p2 = Array(ps_p1.latent_log_sigma)

        x_plot = if vort_panel_use_optimized_latents
            idx = rand(rng, 1:session.n_train, n_plot)
            Array(ps_p1.latent_mu[:, idx])
        else
            randn(rng, T, latent_dim, n_plot)
        end

        gen_omega, _ = eval_decoder_vort(model, N, x_plot, ps_p1, st_p1)
        plot_vorticity_panel(gen_omega, gt_omega, model_dir)

        if model_arch == :fourier
            NDOF_model  = 2 * num_freq - 1
            nfreq_model = num_freq
            gt_oh_re, gt_oh_im = extract_vorticity_spectral(
                eval_snaps, collect(1:n_plot), nfreq_model, NDOF_model, N)
            gt_omega_hat  = complex.(gt_oh_re, gt_oh_im)
            gen_oh_re, gen_oh_im, _ = NN._decode_vort_hat(model, x_plot, ps_p1, st_p1)
            gen_omega_hat = complex.(gen_oh_re, gen_oh_im)
            plot_energy_spectrum(gen_omega_hat, gt_omega_hat, model_dir)
        end

        println("Training complete. Checkpoint and plots saved to $model_dir")

        session = nothing
        GC.gc(true)
        GC.gc(true)
    else
        println("Skipping phase 1: loading decoder weights from $decoder_checkpoint_dir")
        ps_p1, st_p1 = load_checkpoint(decoder_checkpoint_dir)
        init_latent_mu_p2        = ps_p1.latent_mu
        init_latent_log_sigma_p2 = ps_p1.latent_log_sigma
    end

    # ── Phase 2: Neural Operator finetuning ───────────────────────────────────
    if run_phase2
        println("Phase 2: FNO finetuning ($n_fno_layers layers, $fno_channels channels, " *
                "$n_fno_steps steps, n_modes=$n_modes)")

        combined_model, ps_combined, st_combined = UpsamplerWithFNO(
            model, ps_p1, st_p1, N, n_modes, fno_channels, n_fno_layers, n_fno_steps, rng)

        kl_weight_phase2 = fix_latents_phase2 ? 0f0 : kl_weight
        mode2 = if training_mode == :observations
            ObservationsMode(train_snaps, N, batch_size_phase2, n_meas_space, sensor_ci; kl_weight=kl_weight_phase2)
        else
            VorticityMode(train_snaps, N, batch_size_phase2; kl_weight=kl_weight_phase2)
        end

        session2 = TrainingSession(
            rng, combined_model, ps_combined, st_combined, train_snaps, eval_snaps, mode2;
            latent_dim=latent_dim, n_epochs=n_epochs_phase2, eval_every=eval_every_phase2,
            n_slices=n_slices, lr=lr_phase2,
            use_reduce_on_plateau=use_reduce_on_plateau,
            plateau_patience=plateau_patience, plateau_factor=plateau_factor,
            plateau_min_lr=plateau_min_lr,
            fix_thetas=fix_thetas, kl_weight=kl_weight,
            init_latent_mu=init_latent_mu_p2, init_latent_log_sigma=init_latent_log_sigma_p2,
        )

        if freeze_decoder_phase2
            Optimisers.freeze!(session2.tstate.optimizer_state.upsampler)
            println("Decoder frozen: only FNO weights will be updated in phase 2.")
        end
        if fix_latents_phase2
            Optimisers.freeze!(session2.tstate.optimizer_state.latent_mu)
            Optimisers.freeze!(session2.tstate.optimizer_state.latent_log_sigma)
            println("Latents frozen: latent posterior (mu and log_sigma) will not be updated in phase 2.")
        end

        train!(session2)

        phase2_dir = joinpath(model_dir, "phase2")
        config_p2 = Dict{String, Any}(
            "Re" => Re, "N" => N,
            "model_arch" => string(model_arch),
            "T_train" => T_train,
            "batch_size" => batch_size_phase2, "n_epochs" => n_epochs_phase2,
            "eval_every" => eval_every_phase2,
            "latent_dim" => latent_dim,
            "n_slices" => n_slices, "lr" => lr_phase2,
            "use_reduce_on_plateau" => use_reduce_on_plateau,
            "plateau_patience" => plateau_patience, "plateau_factor" => plateau_factor,
            "plateau_min_lr" => plateau_min_lr,
            "fix_thetas" => fix_thetas,
            "kl_weight" => kl_weight,
            "training_mode" => string(training_mode),
            "n_fno_steps" => n_fno_steps, "n_modes" => n_modes,
            "fno_channels" => fno_channels, "n_fno_layers" => n_fno_layers,
            "freeze_decoder_phase2" => freeze_decoder_phase2,
            "fix_latents_phase2" => fix_latents_phase2,
            "decoder_checkpoint_dir" => isnothing(decoder_checkpoint_dir) ? "" : decoder_checkpoint_dir,
        )
        save_checkpoint(phase2_dir,
            session2.tstate.parameters, session2.tstate.states,
            session2.train_losses, session2.eval_swds, session2.eval_epochs,
            config_p2)
        println("Phase 2 checkpoint saved to $phase2_dir")

        !isempty(session2.train_losses) && plot_train_loss_curve(session2.train_losses, phase2_dir)
        !isempty(session2.eval_swds)    && plot_eval_swd_curve(session2.eval_swds, session2.eval_epochs, phase2_dir)

        cpu_p2    = Lux.cpu_device()
        ps_p2_cpu = cpu_p2(session2.tstate.parameters)
        st_p2_cpu = cpu_p2(session2.tstate.states)

        x_plot_p2 = if vort_panel_use_optimized_latents
            idx = rand(rng, 1:session2.n_train, n_plot)
            Array(ps_p2_cpu.latent_mu[:, idx])
        else
            randn(rng, T, latent_dim, n_plot)
        end

        gen_omega_p2, _ = eval_decoder_vort(combined_model, N, x_plot_p2, ps_p2_cpu, st_p2_cpu)
        plot_vorticity_panel(gen_omega_p2, gt_omega, phase2_dir)

        println("Phase 2 training complete. Results saved to $phase2_dir")
    end
end

main()
