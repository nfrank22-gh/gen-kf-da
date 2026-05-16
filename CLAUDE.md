# gen_DA_project

Julia codebase for generating training data and training a generative model for data assimilation (DA). Solves 2D Kolmogorov-forced Navier-Stokes on a GPU using a pseudo-spectral method, tracks passive tracer particles, and saves trajectories for downstream model training.

## Running scripts

Always activate the project environment:

```
julia --project=. scripts/gen_data.jl
julia --project=. scripts/train_ctrl.jl
julia --project=. test/runtests.jl
```

## Package structure

```
src/
  Gen_DA.jl              # top-level module; includes Solver and NN submodules
  Gen_DA/
    Solver.jl            # submodule: NS solver + particle tracking
    Solver/
      rhs.jl             # KfRhs struct, spectral operators, Kolmogorov forcing
      update_fn.jl       # 5-stage IMEX Runge-Kutta timestepper (kf_step, kf_step_particles)
      integrate.jl       # integrate() with/without particle tracking; random ICs
      particles.jl       # bilinear_interp_periodic (pure array gather), tracer_substep
      solver_plotting.jl # plot_vorticity, animate_particles (CairoMakie)
    spectral_grid.jl     # SpectralGrid struct: N, kx (1×N), ky ((N÷2+1)×1), lap (precomputed,
                         #   DC regularised); single source for frequency layout used by both
                         #   Solver and NN
    NN.jl                # submodule: generative model + training pipeline
    NN/
      model.jl           # VortFourierDecoder <: Lux.AbstractLuxLayer (stores SpectralGrid);
                         #   eval_decoder_vort, eval_decoder_vel, spectral_pad;
                         #   _decode_vort_hat is the private shared spectral primitive
      loss_fn.jl         # sliced_wasserstein (physical-space),
                         #   loss_fn, loss_fn_vort_state (both arch-agnostic)
      fno.jl             # FNOLayer (spectral channel-mix + bypass Conv + gelu);
                         #   FourierNeuralOperator (lift → layers → project, residual skip);
                         #   UpsamplerWithFNO (upsampler + FNO × n_fno_steps, phase-2 model);
                         #   eval_decoder_vort / eval_decoder_vel dispatches for UpsamplerWithFNO
      data_pipeline.jl   # DataPipeline submodule: load_trajectory, split_trajectory,
                         #   make_sensor_array, batch_partition, extract_observations
                         #   (takes SpectralGrid), extract_vorticity_spectral
      checkpoint.jl      # Checkpoint submodule: save_checkpoint (weights.jld2, train_log.jld2,
                         #   config.json); no optimizer state saved
      training_utils.jl  # build_optimizer (Adam); ReduceOnPlateau LR scheduler + step!
      training_plots.jl  # TrainingPlots submodule: plot_train_loss_curve, plot_eval_swd_curve,
                         #   plot_vorticity_panel, plot_energy_spectrum
      batch_sampler.jl   # BatchSampler: owns latent-vector and SWD-projection sampling;
                         #   sample_epoch! (returns CPU arrays), get_batch (returns x, thetas, cols)
      training_session.jl # TrainingSession: owns model, tstate, data, mode-specific closures,
                          #   BatchSampler, compiled eval; train!(session) is the training loop;
                          #   phase-2 uses UpsamplerWithFNO as model with VorticityMode or ObservationsMode
scripts/
  gen_data.jl            # spin-up + integrate + save to data/particles/ or data/no_particles/
  train_ctrl.jl          # wiring script: configure hyperparameters → build TrainingSession →
                         #   train! → checkpoint → diagnostic plots
test/
  runtests.jl            # test entry point
  test_data.jl           # CPU tests for DataPipeline
  test_model.jl          # CPU tests for spectral_pad
  test_checkpoint.jl     # CPU tests for Checkpoint (file creation, round-trip, config keys)
  test_train.jl          # CPU tests for build_optimizer
  test_batch_sampler.jl  # CPU tests for BatchSampler (shapes, caching, resampling)
  test_conv_decoder.jl   # CPU tests for ConvDecoder (output shapes, norm_type variants)
  test_relaxation.jl     # CPU tests for relax (shape, identity at 0 steps) and
                         #   loss_fn_relaxation (forward pass, finite/non-negative loss)
```

## Data output

`gen_data.jl` writes to `data/particles/Re{Re}_N{N}_npart{npart}/trajectory.jld2` (and optionally `animation.mp4`). Fields saved: `trajectory`, `xp_traj`, `yp_traj`, `Re`, `n`, `N`, `dt`, `save_every`, `npart`.

`train_ctrl.jl` writes phase-1 output to `<traj_dir>/model/` and phase-2 output to `<traj_dir>/model/phase2/`. Each directory contains:
- `weights.jld2` — model parameters `ps` and state `st` (CPU arrays)
- `train_log.jld2` — `train_losses` (per-epoch), `eval_swds` (per-eval), `eval_epochs`
- `config.json` — all hyperparameters + sensor locations as integer pairs
- `loss_curve.png`, `eval_swd_curve.png`, `vorticity_panel.png` (and `energy_spectrum.png` for phase 1 with `:fourier` arch)

## Key parameters (train_ctrl.jl `main()`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `T_train` | 8000.0 | Training horizon cutoff (solver time units) |
| `n_meas_space` | 50 | Sensor locations (observations mode only) |
| `batch_size` | 400 | Snapshots per training batch |
| `n_epochs` | 100 | Total training epochs |
| `eval_every` | 10 | Eval SWD computed every N epochs |
| `num_freq` | 8 | Spectral resolution (NDOF = 2·num_freq−1 = 15) |
| `latent_dim` | 200 | Latent vector dimension |
| `layers` | [200, 256, 512] | MLP hidden layer widths (first entry = latent_dim) |
| `n_slices` | 10000 | SWD projection directions per batch |
| `lr` | 0.1 | Adam learning rate |
| `use_reduce_on_plateau` | true | Enable ReduceOnPlateau LR scheduler |
| `plateau_patience` | 10 | Epochs without improvement before LR reduction |
| `plateau_factor` | 0.5 | LR multiplier on plateau |
| `plateau_min_lr` | 1e-6 | Minimum LR floor |
| `kl_weight` | 1e-3 | Weight on L2 latent regularization (KL to N(0,I)) |
| `fix_thetas` | true | Fix SWD projection directions for entire training run |
| `training_mode` | `:vorticity` | `:observations` (sparse velocity) or `:vorticity` (full spectral) |
| `run_phase2` | false | Enable phase-2 neural operator finetuning after phase 1 |
| `n_fno_steps` | 3 | Autoregressive FNO applications per sample in phase 2 |
| `n_modes` | 8 | FNO spectral truncation (modes per x/y direction) |
| `fno_channels` | 32 | FNO hidden channel width |
| `n_fno_layers` | 4 | Number of FNO layers |
| `n_epochs_phase2` | 50 | Training epochs for phase 2 |
| `eval_every_phase2` | 10 | Eval SWD frequency for phase 2 |
| `lr_phase2` | 1e-4 | Adam learning rate for phase 2 (trains both upsampler and FNO) |
| `freeze_decoder_phase2` | false | If true, only FNO weights are updated in phase 2; upsampler is optimizer-frozen via `Optimisers.freeze!` (see ADR 0004) |
| `fix_latents_phase2` | false | If true, latent matrix is frozen in phase 2 via `Optimisers.freeze!` |
| `decoder_checkpoint_dir` | `nothing` | If set to a directory path, load decoder weights from that checkpoint and skip phase-1 training entirely; architecture hyperparameters must match |

## Key parameters (gen_data.jl `main()`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `Re` | 100 | Reynolds number |
| `n` | 4 | Kolmogorov forcing wavenumber |
| `N` | 128 | Grid resolution (N×N) |
| `dt` | 0.01 | Timestep |
| `T_spinup` | 50.0 | Spin-up time before recording |
| `T_data` | 1000 | Integration time to record |
| `save_every` | 10 | Save snapshot every N steps |
| `npart` | 40 | Number of passive tracer particles |

## Reactant / GPU backend

The solver uses [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (XLA backend) rather than CUDA.jl directly. CUDA.jl is not a dependency.

- GPU backend is selected in `gen_data.jl` and `train_ctrl.jl` via `Reactant.set_default_backend("cuda")`.
- `KfRhs` is constructed on CPU in `gen_data.jl` and passed directly to `@compile`, which embeds its array values as XLA constants in the compiled function. Phase-2 no longer uses the KF solver.
- Step functions are compiled once with `@compile` before the integration loop; the compiled callables are passed into `integrate()`.
- Saving uses `irfft(Array(omega_hat), N)` — an explicit device→CPU transfer every `save_every` steps.

### Reactant-specific constraints

- **Type conversions**: use `floor.(Int32, x)` to convert `Float32` traced arrays to `Int32` (Reactant defines `Base.floor(::Type{T<:Integer}, ::TracedRNumber{<:AbstractFloat})`). `Int32.(x)` and `Float32.(x)` do not work on traced arrays.
- **Float32 discipline**: all RK coefficients (`alpha`, `beta`, `gamma`) and literals in `update_fn.jl` must be `Float32` to avoid type promotion to `Float64` inside compiled functions.
- **`KfRhs` type parameters**: the struct uses unconstrained `T`, `RA`, and `CA` type parameters (no `<:AbstractFloat` or `<:AbstractArray` bounds) so that Reactant can create traced versions (e.g., `KfRhs{TracedRNumber{Float32}, TracedRArray{...}, ...}`) when `KfRhs` is captured in a `Reactant.@trace` while-loop state.

## Training AD notes

- **Enzyme cannot trace FFTW on CPU**: `loss_fn_vort_state` and FNO layers (both call `irfft` / `rfft`) cannot be tested with AutoEnzyme on CPU. Forward-pass-only CPU tests are fine. Production `train_ctrl.jl` uses `AutoEnzyme()` with Reactant arrays where XLA handles FFTs natively.
- **Optimizer state excluded from checkpoints**: `save_checkpoint` saves `ps` and `st` only. Resuming from a checkpoint requires re-initializing the optimizer.
- **Lux re-exports**: `AutoEnzyme` (from `ADTypes`) and `reactant_device` (from `MLDataDevices`) are not in their originating packages' public namespaces — use `Lux.AutoEnzyme()` and `Lux.reactant_device()`. Both are re-exported by `Lux`.

## Numerics

The solver uses a 5-stage low-storage IMEX Runge-Kutta scheme. Nonlinear advection is treated explicitly (with 2/3-rule spectral dealiasing); viscous diffusion is treated implicitly. Arrays on device are `ConcretePJRTArray` (Reactant's PJRT-backed type). The spectral layout is `(N÷2+1, N)` — `rfft` along dimension 1 (y), full FFT along dimension 2 (x).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`nfrank22-gh/gen-kf-da`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
