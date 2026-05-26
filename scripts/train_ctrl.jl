using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Gen_DA.NN.TrainingPlots
using Reactant, Random, Lux, LinearAlgebra

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
    random_sensors = false
    n_meas_space  = Inf         # only used when training_mode == :observations
    batch_size    = 2000
    n_epochs      = 5000
    eval_every    = 100
    latent_dim    = 100
    n_slices      = 5000
    lr                    = 1f-2
    # :reduce_on_plateau, :cosine_annealing, :cosine_annealing_warm_restarts, or :none
    lr_scheduler          = :cosine_annealing
    plateau_patience      = 20
    plateau_factor        = 0.5f0
    plateau_min_lr        = 1f-6
    cosine_T_max          = n_epochs
    cosine_eta_min        = 1f-4
    cosine_T_mult         = 1         # cycle length multiplier for warm restarts (ignored otherwise)
    fix_thetas            = false
    kl_weight             = Float32(0.1) # beta-VAE weight on KL(N(mu,sigma²) || N(0,I))
    h2_weight             = Float32(1f-9)       # H2 vorticity regularization weight (0 = disabled)
    latent_lr_multiplier  = Float32(.1)   # LR multiplier for latent tables (VorticityMode only)
    # :observations — sparse velocity at sensor locations (production)
    # :vorticity    — full spectral vorticity fields (testing simplification)
    training_mode = :vorticity
    recon_loss      = :sinkhorn      # :swd, :sinkhorn, or :mse
    sinkhorn_eps    = Float32(.01)     # ε for Sinkhorn divergence (ignored when recon_loss != :sinkhorn)
    sinkhorn_n_iter = 300       # Sinkhorn iterations   (ignored when recon_loss != :sinkhorn)
    sinkhorn_metric = :sq_l2   # :sq_l2, :cosine, :lr_mahalanobis, or :rand_lr_mahalanobis  (ignored when recon_loss != :sinkhorn)
    mahalanobis_rank = 500  # Int rank r for :lr_mahalanobis; must be set when sinkhorn_metric = :lr_mahalanobis
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
        conv_channels     = [16, 8, 4, 2]   # output channels per block (length = n_blocks)
        n_convs_per_block   = 2             # DenseNet convolutions per UpsampleBlock
        kernel_sizes        = [3, 5, 7, 9]    # kernel size per block (length = n_blocks)
        spectral_modes      = [4, 4, 4, 4]    # FNO k_max per UpsampleBlock
        tail_kernel         = 9
        tail_spectral_modes = 3          # FNO k_max for tail convs
        use_spectral        = false         # false → pure CircConv, no rfft branch
        upsample_mode       = :spectral    # :spectral (zero-pad rfft), :conv_transpose (learned), or :nearest (nearest-neighbor)
        upsample_kernel     = 2            # kernel size for :conv_transpose upsample (even recommended)
        use_ge              = true        # true → GE-θ+ spatial channel attention in each UpsampleBlock
        ge_kernel_sizes     = [5, 9, 13, 15]   # depth-wise gather kernel per block (length = n_blocks)
        ge_reduction        = 4           # bottleneck reduction ratio for GE excite FC
        use_antialias       = false        # true → spectral upsample/downsample wraps each activation
        norm_type           = :batch      # :batch (BatchNorm) or :instance (InstanceNorm)
        # Constraint: 2·k_base·2^n_blocks == N_conv; N ÷ N_conv must be a power of 2.
        # k_base=4, n_blocks=3: 2·4·8 = 64 = N_conv for N=128 with one extra spectral upsample.
        N_conv            = div(N, 1)
        model, ps, st = ConvDecoder(latent_dim, fc_hidden, k_base, init_channels,
                                    conv_channels, n_convs_per_block, kernel_sizes, tail_kernel,
                                    act, N_conv, N, rng, T;
                                    spectral_modes=spectral_modes,
                                    tail_spectral_modes=tail_spectral_modes,
                                    use_spectral=use_spectral,
                                    upsample_mode=upsample_mode,
                                    upsample_kernel=upsample_kernel,
                                    use_ge=use_ge,
                                    ge_kernel_sizes=ge_kernel_sizes,
                                    ge_reduction=ge_reduction,
                                    use_antialias=use_antialias,
                                    norm_type=norm_type)
    
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

    # Standardize using training statistics only; apply same stats to eval.
    train_snaps, mean_field, std_field = standardize_snapshots(train_snaps)
    eval_snaps = apply_standardization(eval_snaps, mean_field, std_field)
    println("Snapshots standardized (per-pixel z-score from training data).")

    sensor_ci = (!random_sensors && training_mode == :observations && !isinf(n_meas_space)) ? make_sensor_array(N, n_meas_space, rng) : nothing

    # ── Mahalanobis basis (PCA of standardized training features) ─────────────
    mahalanobis_L = nothing
    if sinkhorn_metric == :lr_mahalanobis
        mahalanobis_rank === nothing && error(
            "mahalanobis_rank must be set when sinkhorn_metric = :lr_mahalanobis")
        n_pca = min(size(train_snaps, 3), 2000)
        pca_idx = randperm(rng, size(train_snaps, 3))[1:n_pca]
        if training_mode == :vorticity
            flat = reshape(train_snaps[:, :, pca_idx], N * N, n_pca)
        else
            @assert !random_sensors "sinkhorn_metric = :lr_mahalanobis is not supported with random_sensors = true"
            pca_grid = SpectralGrid(N)
            u_pca, v_pca = if isinf(n_meas_space)
                DataPipeline.extract_full_velocity(train_snaps, pca_idx, pca_grid)
            else
                DataPipeline.extract_observations(train_snaps, pca_idx, sensor_ci, pca_grid)
            end
            flat = vcat(u_pca, v_pca)
        end
        U_pca = svd(flat; full=false).U
        mahalanobis_L = Matrix{Float32}(U_pca[:, 1:mahalanobis_rank])
        println("Mahalanobis basis computed: rank=$(mahalanobis_rank), feature_dim=$(size(mahalanobis_L,1))")
    end

    # ── Eval plot data ─────────────────────────────────────────────────────────
    n_plot   = 4
    # true  → pick n_plot random columns from the trained latent matrix
    # false → draw fresh samples from N(0,I)
    vort_panel_use_optimized_latents = false
    gt_idx   = rand(rng, 1:size(eval_snaps, 3), n_plot)
    gt_omega = eval_snaps[:, :, gt_idx]

    # ── Training ───────────────────────────────────────────────────────────────
    init_latent_mu        = nothing
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
                         h2_weight=h2_weight,
                         recon_loss=recon_loss,
                         sinkhorn_eps=sinkhorn_eps,
                         sinkhorn_n_iter=sinkhorn_n_iter,
                         sinkhorn_metric=sinkhorn_metric,
                         mahalanobis_L=mahalanobis_L,
                         mahalanobis_rank=mahalanobis_rank)
    else
        VorticityMode(train_snaps, N, batch_size;
                      h2_weight=h2_weight,
                      recon_loss=recon_loss,
                      sinkhorn_eps=sinkhorn_eps,
                      sinkhorn_n_iter=sinkhorn_n_iter,
                      sinkhorn_metric=sinkhorn_metric,
                      mahalanobis_L=mahalanobis_L,
                      mahalanobis_rank=mahalanobis_rank,
                      rng=rng)
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
        h2_weight=h2_weight,
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
        "h2_weight" => h2_weight,
        "latent_lr_multiplier" => latent_lr_multiplier,
        "training_mode" => string(training_mode),
        "recon_loss" => string(recon_loss),
        "sinkhorn_eps" => sinkhorn_eps,
        "sinkhorn_n_iter" => sinkhorn_n_iter,
        "sinkhorn_metric" => string(sinkhorn_metric),
        "mahalanobis_rank" => mahalanobis_rank,
        "random_sensors" => random_sensors,
        "mean_field" => vec(mean_field),
        "std_field"  => vec(std_field),
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
        config["use_spectral"]          = use_spectral
        config["norm_type"]             = string(norm_type)
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

    gen_omega_std, _ = eval_decoder_vort(model, N, x_plot, ps_cpu, Lux.testmode(st_cpu))
    gen_omega = gen_omega_std .* std_field .+ mean_field
    gt_omega  = gt_omega      .* std_field .+ mean_field
    plot_vorticity_panel(gen_omega, gt_omega, model_dir)

    if model_arch == :fourier
        dec, dec_ps, dec_st = model, ps_cpu, st_cpu
        NDOF_model  = 2 * num_freq - 1
        nfreq_model = num_freq
        gt_oh_re, gt_oh_im = extract_vorticity_spectral(
            eval_snaps, collect(1:n_plot), nfreq_model, NDOF_model, N)
        gt_omega_hat  = complex.(gt_oh_re, gt_oh_im)
        gen_omega_hat, _ = NN._decode_omega_hat(dec, x_plot, dec_ps, dec_st)
        plot_energy_spectrum(gen_omega_hat, gt_omega_hat, model_dir)
    end

    println("Training complete. Checkpoint and plots saved to $model_dir")
end

main()
