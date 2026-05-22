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
    # ── Sensor array (observations mode only) ─────────────────────────────────
    # random_sensors = true: one unique layout per training snapshot, fixed for the run.
    # random_sensors = false: one shared layout for all snapshots.
    random_sensors = true
    n_meas_space  = Inf         # only used when training_mode == :observations
    batch_size    = 801
    n_epochs      = 2000
    eval_every    = 100
    latent_dim    = 200
    n_slices      = 5000
    lr                    = 1f-3
    # :reduce_on_plateau, :cosine_annealing, :cosine_annealing_warm_restarts, or :none
    lr_scheduler          = :cosine_annealing
    plateau_patience      = 20
    plateau_factor        = 0.5f0
    plateau_min_lr        = 1f-6
    cosine_T_max          = n_epochs
    cosine_eta_min        = 1f-5
    cosine_T_mult         = 1         # cycle length multiplier for warm restarts (ignored otherwise)
    fix_thetas            = false
    kl_weight             = Float32(0.0001) # beta-VAE weight on KL(N(mu,sigma²) || N(0,I))
    latent_lr_multiplier  = Float32(.1)   # LR multiplier for latent tables (VorticityMode only)
    # :observations — sparse velocity at sensor locations (production)
    # :vorticity    — full spectral vorticity fields (testing simplification)
    training_mode = :observations
    recon_loss      = :sinkhorn      # :swd, :sinkhorn, or :mse
    sinkhorn_eps    = 0.1f0     # ε for Sinkhorn divergence (ignored when recon_loss != :sinkhorn)
    sinkhorn_n_iter = 400       # Sinkhorn iterations   (ignored when recon_loss != :sinkhorn)
    resume_phase1 = false   # if true, load weights from model_dir and continue training

    # ── Model ─────────────────────────────────────────────────────────────────
    # :fourier — StreamFourierDecoder (MLP → stream function ψ̂ → derived u,v)
    # :conv    — ConvDecoder (FC → spectral upsample blocks → stream function ψ)
    model_arch = :conv

    rng = Xoshiro(3)

    if model_arch == :fourier
        num_freq = 8                          # spectral resolution; NDOF = 2·num_freq − 1
        layers   = [latent_dim, 256, 512]     # MLP hidden widths (first entry must equal latent_dim)
        model, ps, st = StreamFourierDecoder(layers, num_freq, rng, T)

    elseif model_arch == :conv
        act               = gelu
        fc_hidden         = [256]         # hidden layers for FC latent → Fourier coefficients
        k_base            = 4             # wavenumber cutoff; irfft output is (2·k_base)×(2·k_base)
        init_channels     = 4            # C: initial channel count (Fourier base output)
        conv_channels     = [16, 8, 4]   # output channels per block (length = n_blocks)
        n_convs_per_block   = 4             # DenseNet convolutions per UpsampleBlock
        kernel_sizes        = [3, 5, 7]    # kernel size per block (length = n_blocks)
        spectral_modes      = [4, 4, 6]    # FNO k_max per UpsampleBlock
        tail_kernel         = 9
        tail_spectral_modes = 6            # FNO k_max for tail convs
        # Constraint: 2·k_base·2^n_blocks == N_conv; N ÷ N_conv must be a power of 2.
        # k_base=4, n_blocks=3: 2·4·8 = 64 = N_conv for N=128 with one extra spectral upsample.
        N_conv            = div(N, 2)
        model, ps, st = ConvDecoder(latent_dim, fc_hidden, k_base, init_channels,
                                    conv_channels, n_convs_per_block, kernel_sizes, tail_kernel,
                                    act, N_conv, N, rng, T;
                                    spectral_modes=spectral_modes,
                                    tail_spectral_modes=tail_spectral_modes)
    
    end

    # ── Data ──────────────────────────────────────────────────────────────────
    traj_path = "data/no_particles/Re=$(Re)_N=$(N)_dt=$(data_dt)_T=$(T_data)/trajectory.jld2"
    model_dir = joinpath(dirname(traj_path), "model")
    traj_data = load_trajectory(traj_path)
    n_forcing = traj_data.n   # Kolmogorov forcing wavenumber from data generation

    all_snaps = reduce_trajectory(traj_data.trajectory, n_forcing)
    train_snaps, eval_snaps = split_trajectory(
        all_snaps, traj_data.dt, traj_data.save_every, T_train)
    println("Training on $(size(train_snaps, 3)) snapshots, eval on $(size(eval_snaps, 3)) snapshots")

    sensor_ci = (!random_sensors && training_mode == :observations && !isinf(n_meas_space)) ? make_sensor_array(N, n_meas_space, rng) : nothing

    # ── Eval plot data ─────────────────────────────────────────────────────────
    n_plot   = 4
    # true  → pick n_plot random columns from the trained latent matrix
    # false → draw fresh samples from N(0,I)
    vort_panel_use_optimized_latents = false
    gt_idx   = rand(rng, 1:size(eval_snaps, 3), n_plot)
    gt_omega = eval_snaps[:, :, gt_idx]

    # ── Training ───────────────────────────────────────────────────────────────
  init_latent_mu        = Float32(0)
    init_latent_log_sigma = Float32(-8)
    if resume_phase1
        println("Resuming: loading weights from $model_dir")
        loaded_ps, loaded_st = load_checkpoint(model_dir)
        ps = loaded_ps
        st = loaded_st
        init_latent_mu        = ps.latent_mu
        init_latent_log_sigma = ps.latent_log_sigma
    end

    mode = if training_mode == :observations
        ObservationsMode(train_snaps, N, batch_size, n_meas_space, sensor_ci;
                         random_sensors=random_sensors, rng=rng,
                         recon_loss=recon_loss,
                         sinkhorn_eps=sinkhorn_eps,
                         sinkhorn_n_iter=sinkhorn_n_iter)
    else
        VorticityMode(train_snaps, N, batch_size;
                      recon_loss=recon_loss,
                      sinkhorn_eps=sinkhorn_eps,
                      sinkhorn_n_iter=sinkhorn_n_iter)
    end

    session = TrainingSession(
        rng, model, ps, st, train_snaps, eval_snaps, mode;
        latent_dim=latent_dim, n_epochs=n_epochs, eval_every=eval_every,
        n_slices=n_slices, lr=lr,
        lr_scheduler=lr_scheduler,
        plateau_patience=plateau_patience, plateau_factor=plateau_factor,
        plateau_min_lr=plateau_min_lr,
        cosine_T_max=cosine_T_max, cosine_eta_min=cosine_eta_min, cosine_T_mult=cosine_T_mult,
        fix_thetas=fix_thetas,
        kl_weight=kl_weight,
        latent_lr_multiplier=latent_lr_multiplier,
        init_latent_mu=init_latent_mu,
        init_latent_log_sigma=init_latent_log_sigma,
    )

    train!(session)

    config = Dict{String, Any}(
        "Re" => Re, "N" => N,
        "model_arch" => string(model_arch),
        "traj_path" => traj_path,
        "n_train" => size(train_snaps, 3),
        "T_train" => T_train, "n_meas_space" => isinf(n_meas_space) ? "all" : n_meas_space,
        "batch_size" => batch_size, "n_epochs" => n_epochs,
        "eval_every" => eval_every,
        "latent_dim" => latent_dim,
        "n_slices" => n_slices, "lr" => lr,
        "lr_scheduler" => string(lr_scheduler),
        "plateau_patience" => plateau_patience, "plateau_factor" => plateau_factor,
        "plateau_min_lr" => plateau_min_lr,
        "cosine_T_max" => cosine_T_max, "cosine_eta_min" => cosine_eta_min, "cosine_T_mult" => cosine_T_mult,
        "fix_thetas" => fix_thetas,
        "kl_weight" => kl_weight,
        "latent_lr_multiplier" => latent_lr_multiplier,
        "training_mode" => string(training_mode),
        "recon_loss" => string(recon_loss),
        "sinkhorn_eps" => sinkhorn_eps,
        "sinkhorn_n_iter" => sinkhorn_n_iter,
        "random_sensors" => random_sensors,
    )
    if model_arch == :fourier
        config["num_freq"] = num_freq
        config["layers"]   = layers
    elseif model_arch == :conv
        config["k_base"]             = k_base
        config["N_conv"]             = N_conv
        config["fc_hidden"]          = fc_hidden
        config["init_channels"]      = init_channels
        config["conv_channels"]      = conv_channels
        config["n_convs_per_block"]  = n_convs_per_block
        config["kernel_sizes"]          = kernel_sizes
        config["spectral_modes"]        = spectral_modes
        config["tail_kernel"]           = tail_kernel
        config["tail_spectral_modes"]   = tail_spectral_modes
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
    ps_cpu = cpu(session.tstate.parameters)
    st_cpu = cpu(session.tstate.states)

    x_plot = if vort_panel_use_optimized_latents
        idx = rand(rng, 1:session.n_train, n_plot)
        Array(ps_cpu.latent_mu[:, idx])
    else
        randn(rng, T, latent_dim, n_plot)
    end

    gen_omega, _ = eval_decoder_vort(model, N, x_plot, ps_cpu, st_cpu)
    plot_vorticity_panel(gen_omega, gt_omega, model_dir)

    if model_arch == :fourier
        dec, dec_ps, dec_st = model, ps_cpu, st_cpu
        NDOF_model  = 2 * num_freq - 1
        nfreq_model = num_freq
        gt_oh_re, gt_oh_im = extract_vorticity_spectral(
            eval_snaps, collect(1:n_plot), nfreq_model, NDOF_model, N)
        gt_omega_hat  = complex.(gt_oh_re, gt_oh_im)
        psi_hat, _ = NN._decode_psi_hat(dec, x_plot, dec_ps, dec_st)
        gen_omega_hat = .-dec.grid.lap .* psi_hat   # |k|² · ψ̂
        plot_energy_spectrum(gen_omega_hat, gt_omega_hat, model_dir)
    end

    println("Training complete. Checkpoint and plots saved to $model_dir")
end

main()
