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
      loss_fn.jl         # sliced_wasserstein, sliced_wasserstein_spectral,
                         #   loss_fn, loss_fn_vort, loss_fn_vort_state
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
                          #   BatchSampler, compiled eval; train!(session) is the training loop
scripts/
  gen_data.jl            # spin-up + integrate + save to data/particles/ or data/no_particles/
  train_ctrl.jl          # wiring script: configure hyperparameters → build TrainingSession →
                         #   train! → checkpoint → diagnostic plots
test/
  runtests.jl            # test entry point
  test_data.jl           # CPU tests for DataPipeline
  test_loss_fn.jl        # CPU tests for sliced_wasserstein_spectral
  test_model.jl          # CPU tests for spectral_pad
  test_checkpoint.jl     # CPU tests for Checkpoint (file creation, round-trip, config keys)
  test_train.jl          # CPU tests for build_optimizer
  test_batch_sampler.jl  # CPU tests for BatchSampler (shapes, caching, resampling)
```

## Data output

`gen_data.jl` writes to `data/particles/Re{Re}_N{N}_npart{npart}/trajectory.jld2` (and optionally `animation.mp4`). Fields saved: `trajectory`, `xp_traj`, `yp_traj`, `Re`, `n`, `N`, `dt`, `save_every`, `npart`.

`train_ctrl.jl` writes to `<traj_dir>/model/`:
- `weights.jld2` — model parameters `ps` and state `st` (CPU arrays)
- `train_log.jld2` — `train_losses` (per-epoch), `eval_swds` (per-eval), `eval_epochs`
- `config.json` — all hyperparameters + sensor locations as integer pairs
- `loss_curve.png`, `eval_swd_curve.png`, `vorticity_panel.png`, `energy_spectrum.png`

## Key parameters (train_ctrl.jl `main()`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `T_train` | 8000.0 | Training horizon cutoff (solver time units) |
| `n_meas_space` | 50 | Sensor locations (observations mode only) |
| `batch_size` | 400 | Snapshots per training batch |
| `n_epochs` | 1000 | Total training epochs |
| `eval_every` | 10 | Eval SWD computed every N epochs |
| `num_freq` | 8 | Model spectral resolution (NDOF = 2·num_freq−1 = 15) |
| `latent_dim` | 200 | Latent vector dimension |
| `layers` | [200, 256, 512] | MLP hidden layer widths |
| `n_slices` | 10000 | SWD projection directions per batch |
| `lr` | 0.1 | Adam learning rate |
| `use_reduce_on_plateau` | true | Enable ReduceOnPlateau LR scheduler |
| `plateau_patience` | 10 | Epochs without improvement before LR reduction |
| `plateau_factor` | 0.5 | LR multiplier on plateau |
| `plateau_min_lr` | 1e-6 | Minimum LR floor |
| `fix_x` | true | Fix latent samples for entire training run |
| `fix_thetas` | true | Fix SWD projection directions for entire training run |
| `training_mode` | `:vorticity` | `:observations` (sparse velocity) or `:vorticity` (full spectral) |

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

- GPU backend is selected in `gen_data.jl` via `Reactant.set_default_backend("gpu")`.
- `KfRhs` is constructed on CPU then moved to device with `to_reactant(rhs)` (defined in `gen_data.jl`).
- Step functions are compiled once with `@compile` before the integration loop; the compiled callables are passed into `integrate()`.
- Saving uses `irfft(Array(omega_hat), N)` — an explicit device→CPU transfer every `save_every` steps.

### Reactant-specific constraints

- **Type conversions**: use `floor.(Int32, x)` to convert `Float32` traced arrays to `Int32` (Reactant defines `Base.floor(::Type{T<:Integer}, ::TracedRNumber{<:AbstractFloat})`). `Int32.(x)` and `Float32.(x)` do not work on traced arrays.
- **Float32 discipline**: all RK coefficients (`alpha`, `beta`, `gamma`) and literals in `update_fn.jl` must be `Float32` to avoid type promotion to `Float64` inside compiled functions.
- **`KfRhs` type parameters**: the struct uses unconstrained `RA` and `CA` type parameters (no `AbstractArray` bounds) so that `TracedRArray` fields are accepted during tracing.

## Training AD notes

- **CPU tests use AutoEnzyme via loss_fn_vort**: `test_train.jl` uses `AutoEnzyme()` with `loss_fn_vort`, which only calls the MLP + `sliced_wasserstein` (no FFTW). Enzyme cannot trace through FFTW plan creation on CPU, so `loss_fn` (which calls `eval_decoder_vel` → `irfft`) cannot be tested this way on CPU. Production `train_ctrl.jl` uses `AutoEnzyme()` with Reactant arrays where XLA handles FFTs natively.
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
