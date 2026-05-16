# Two-phase training with KF solver relaxation as physics refinement

The generative model needs its output distribution to lie on the system's attractor. The upsampling model (VortFourierDecoder / ConvDecoder) can produce physically implausible fields — spectrally consistent but not dynamically consistent — because the SWD loss only enforces distributional matching, not attractor membership.

We chose a two-phase training strategy with optional **relaxation** (running the KF solver for a short `T_relax` from the upsampling model output) as the physics refinement step:

- **Phase 1**: train the upsampling model with SWD on its direct output (existing machinery, unchanged).
- **Phase 2** (optional): compute SWD on the relaxed output; backprop through the short solver rollout into the upsampling model. `freeze_upsampler` controls whether the upsampling model's parameters are updated (always `false` for the KF-solver fine-tuner, which has no learnable parameters of its own). Phase 2 checkpoint goes to `<traj_dir>/model/phase2/`.

**Alternatives considered:**

- *Full backprop through a long solver rollout*: chaotic Lyapunov instabilities could amplify gradients. For Re=100 the Lyapunov timescale is ~3.3 time units, so T_relax up to ~2.0 is well within the gradient-stable regime; longer horizons warrant care.
- *Stop-gradient at the solver boundary*: the upsampling model receives no gradient signal from the relaxed output, requiring a separate loss on its direct output (effectively phase 1 only). Rejected because it gives up on joint training entirely.
- *Two-phase with baked-in phase switching inside `TrainingSession`*: more monolithic; harder to skip phase 2 or checkpoint between phases. Rejected in favour of two separate `TrainingSession` objects constructed sequentially in `train_ctrl.jl`.

**Consequence**: the relaxation duration `T_relax` and timestep `dt_relax` must be short enough that gradients are numerically useful (Lyapunov instabilities grow exponentially past O(0.1) time units). The `KfRhs` used for relaxation is constructed from the same `Re`, `N`, and forcing wavenumber `n` as the data-generating solver — relaxing under different physics would push fields toward the wrong attractor.

## Phase-2 gradient compilation strategy

### Root cause of the original hang

`loss_fn_relaxation` ran a Julia `for` loop inside `Training.single_train_step!(AutoEnzyme(), ...)`. Two approaches were tried:

1. **Unrolled loop** (no `@trace`): Reactant unrolls n_steps into the MLIR graph. For n_steps=200 × 5 RK stages, thousands of ops cause a naming-uniqueness error or indefinitely slow compilation.

2. **`@trace`**: converts the loop to `stablehlo.while`. This fixes the forward compilation but Enzyme-MLIR cannot differentiate through `stablehlo.while` — compilation hangs on the backward pass even for n_steps=10.

### Current decision: manual adjoint with split compilation

The phase-2 training loop bypasses `Training.single_train_step!` for the solver chain entirely. It uses three separately compiled functions plus the standard Enzyme-compiled upsampler VJP:

| Step | Function | Compiled by |
|------|----------|-------------|
| Forward | `_phase2_forward` (upsampler + `relax_and_store`) | `Reactant.@compile` |
| SWD + gradient | `_swd_adjoint_cpu` | CPU, pure Julia |
| Backward | `relax_adj_full` (solver adjoint) | `Reactant.@compile` |
| Param update | VJP inner-product loss | `Training.single_train_step!` / Enzyme |

`relax_and_store` stores `omega_hat` at all n_steps+1 steps as a 4-D array `(N÷2+1, N, batch, n_steps+1)`. The loop is unrolled at trace time (n_steps is a compile-time constant); no Enzyme is involved — just a chain of XLA ops. Compilation is bounded and slow for large n_steps but never hangs.

`relax_adj_full` runs `_kf_step_batched_adjoint` backward through the stored trajectory. Each call re-runs the 5-stage IMEX RK forward from the stored checkpoint to recover intermediate states, then applies the exact spectral adjoint. Enzyme is not invoked for this path.

The upsampler VJP uses the identity: gradient of `sum(omega_gen .* d_omega_gen)` w.r.t. `params` equals `J_upsampler^T * d_omega_gen`. `Training.single_train_step!(AutoEnzyme(), vjp_loss, ...)` compiles and applies this correctly since the upsampler has no solver loops.

**Memory**: the trajectory tensor is `(N÷2+1) × N × batch × (n_steps+1)` complex Float32. For N=128, batch=801, n_steps=200: ~10.8 GB. Fits in the 25 GB BFC allocator with margin; can reduce batch_size if needed.

**Compilation time**: each compiled function unrolls n_steps iterations. For n_steps=200 this takes several minutes on first call; subsequent epochs reuse the cached compiled function.

**Alternative considered and rejected**: `EnzymeRules.augmented_primal` / `reverse` for `relax()`. The tape would need to carry the trajectory as an XLA tuple through Enzyme-MLIR's custom rule mechanism, whose interaction with Reactant's compilation pipeline is uncertain. The manual split-compilation approach is simpler and guaranteed to work.

**Future**: when a neural operator replaces the KF solver as the fine-tuner, `freeze_upsampler = true` becomes meaningful (train only the operator). Full backprop through the operator via Enzyme is expected to be stable since it lacks solver-loop compilation issues.
