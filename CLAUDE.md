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
    NN.jl                # submodule: generative model + data pipeline
    NN/
      model.jl           # VortFourierDecoder <: Lux.AbstractLuxLayer; eval_decoder_vort, eval_decoder_vel
      loss_fn.jl         # sliced_wasserstein, loss_fn
      data_pipeline.jl   # DataPipeline submodule: load_trajectory, split_trajectory,
                         #   make_sensor_array, batch_partition, extract_observations
scripts/
  gen_data.jl            # spin-up + integrate + save to data/particles/ or data/no_particles/
  train_ctrl.jl          # training stub (WIP)
test/
  runtests.jl            # test entry point
  test_data.jl           # CPU tests for DataPipeline
```

## Data output

`gen_data.jl` writes to `data/particles/Re{Re}_N{N}_npart{npart}/trajectory.jld2` (and optionally `animation.mp4`). Fields saved: `trajectory`, `xp_traj`, `yp_traj`, `Re`, `n`, `N`, `dt`, `save_every`, `npart`.

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

## Known issues

- **Lux API**: `AbstractExplicitLayer` was renamed to `AbstractLuxLayer` in Lux v1.x. `model.jl` already uses the new name.

## Numerics

The solver uses a 5-stage low-storage IMEX Runge-Kutta scheme. Nonlinear advection is treated explicitly (with 2/3-rule spectral dealiasing); viscous diffusion is treated implicitly. Arrays on device are `ConcretePJRTArray` (Reactant's PJRT-backed type). The spectral layout is `(N÷2+1, N)` — `rfft` along dimension 1 (y), full FFT along dimension 2 (x).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`nfrank22-gh/gen-kf-da`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
