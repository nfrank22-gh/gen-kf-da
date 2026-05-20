using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Gen_DA.NN.Checkpoint
using Gen_DA.NN.TrainingPlots
import Gen_DA.NN: ReduceOnPlateau, step!
using Reactant, Random, Lux, Statistics
import Lux.Training as Training
import Optimisers

Reactant.set_default_backend("cuda")

# ── Hyperparameters ────────────────────────────────────────────────────────────
const CHECKPOINT_DIR   = "data/no_particles/Re=100_N=128_dt=0.01_T=10000/model"
const SNAP_IDX         = nothing   # nothing → random draw from eval set
const N_MEAS_SPACE     = 200       # sensors for conditioning
const N_INF_STEPS      = 1000      # Adam steps
const LR_INF           = 1f-1
const PLATEAU_PATIENCE = 5
const PLATEAU_FACTOR   = 0.5f0
const PLATEAU_MIN_LR   = 1f-6
const SEED             = 42


function main()
    rng = Xoshiro(SEED)

    config     = load_config(CHECKPOINT_DIR)
    ps, st     = load_checkpoint(CHECKPOINT_DIR)
    N          = Int(config["N"])
    n_train    = Int(config["n_train"])
    latent_dim = Int(config["latent_dim"])
    traj_path  = config["traj_path"]

    T_train    = Float32(config["T_train"])
    traj_data  = load_trajectory(traj_path)
    n_forcing  = traj_data.n
    all_snaps  = reduce_trajectory(traj_data.trajectory, n_forcing)
    _, eval_snaps = split_trajectory(all_snaps, traj_data.dt, traj_data.save_every, T_train)
    n_eval     = size(eval_snaps, 3)
    @assert n_eval > 0 "Checkpoint has no eval snapshots (n_train = $n_train covers the full trajectory)"

    snap_idx = SNAP_IDX !== nothing ? SNAP_IDX : rand(rng, 1:n_eval)
    println("Snapshot DA: conditioning on eval snapshot $snap_idx / $n_eval")

    grid       = SpectralGrid(N)
    sensor_ci  = make_sensor_array(N, N_MEAS_SPACE, rng)
    sensor_lin = Int32.(LinearIndices((N, N))[sensor_ci])

    u_obs, v_obs = extract_observations(eval_snaps, [snap_idx], sensor_ci, grid)
    u_obs = vec(u_obs)
    v_obs = vec(v_obs)

    model = build_model_from_config(config, rng)

    # Strip training-time posterior matrices — not needed for decoder forward pass,
    # and their size (latent_dim × n_train) causes XLA compilation to fail on dead args.
    ps_decoder = Base.structdiff(ps, (latent_mu=nothing, latent_log_sigma=nothing))
    dev    = Lux.reactant_device()
    ps_gpu = dev(ps_decoder)
    st_gpu = Lux.testmode(st) |> dev

    u_obs_ra      = Reactant.to_rarray(u_obs)
    v_obs_ra      = Reactant.to_rarray(v_obs)
    sensor_lin_ra = Reactant.to_rarray(sensor_lin)

    # Merge point-estimate parameters with frozen decoder weights into one parameter tree.
    # This mirrors the phase-1 training pattern (all params in Duplicated) and avoids the
    # lower-enzymexla-ml pass failure that occurs when decoder weights are in Const(data).
    combined_ps = merge(
        (mu = Reactant.to_rarray(zeros(Float32, latent_dim)),),
        ps_gpu,
    )
    tstate = Training.TrainState(model, combined_ps, st_gpu, Optimisers.Adam(LR_INF))

    # Freeze all decoder weight subtrees so only mu gets gradients.
    for key in keys(ps_gpu)
        Optimisers.freeze!(getproperty(tstate.optimizer_state, key))
    end

    function da_loss(model, ps, st, data)
        u_obs_d, v_obs_d, sensor_lin_d = data
        z    = reshape(ps.mu, :, 1)
        u, v, st = eval_decoder_vel(model, N, z, ps, st)
        u_pred   = reshape(u, N * N)[sensor_lin_d]
        v_pred   = reshape(v, N * N)[sensor_lin_d]
        return mean((u_pred .- u_obs_d) .^ 2) + mean((v_pred .- v_obs_d) .^ 2), st, (;)
    end

    sched = ReduceOnPlateau(LR_INF; factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    println("Optimising point estimate ($N_INF_STEPS steps)...")
    data = (u_obs_ra, v_obs_ra, sensor_lin_ra)
    for step in 1:N_INF_STEPS
        _, loss, _, tstate = Training.single_train_step!(
            Lux.AutoEnzyme(), da_loss, data, tstate)
        new_lr = step!(sched, Float32(loss))
        Optimisers.adjust!(tstate.optimizer_state, eta=new_lr)
        step % 10 == 0 && println("  step $step  loss = $loss  lr = $(sched.current_lr)")
    end

    cpu      = Lux.cpu_device()
    ps_cpu   = cpu(tstate.parameters)
    z_est    = reshape(ps_cpu.mu, :, 1)
    gen_omega, _ = eval_decoder_vort(model, N, z_est, ps_cpu, cpu(tstate.states))
    est_omega = gen_omega[:, :, 1]
    gt_omega  = eval_snaps[:, :, snap_idx]

    out_dir = joinpath(CHECKPOINT_DIR, "snapshot_da", "snap_$(snap_idx)")
    plot_da_point_estimate(est_omega, gt_omega, sensor_ci, out_dir)
    println("Output saved to $out_dir")
end

main()
