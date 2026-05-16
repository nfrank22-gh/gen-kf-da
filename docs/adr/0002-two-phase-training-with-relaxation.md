# Two-phase training with KF solver relaxation as physics refinement

The generative model needs its output distribution to lie on the system's attractor. The upsampling model (VortFourierDecoder / ConvDecoder) can produce physically implausible fields — spectrally consistent but not dynamically consistent — because the SWD loss only enforces distributional matching, not attractor membership.

We chose a two-phase training strategy with optional **relaxation** (running the KF solver for a short `T_relax` from the upsampling model output) as the physics refinement step:

- **Phase 1**: train the upsampling model with SWD on its direct output (existing machinery, unchanged).
- **Phase 2** (optional): compute SWD on the relaxed output; backprop through the short solver rollout into the upsampling model. `freeze_upsampler` controls whether the upsampling model's parameters are updated (always `false` for the KF-solver fine-tuner, which has no learnable parameters of its own). Phase 2 checkpoint goes to `<traj_dir>/model/phase2/`.

**Alternatives considered:**

- *Full backprop through a long solver rollout*: chaotic Lyapunov instabilities make gradients numerically useless past a few steps. Rejected for any `T_relax` beyond O(0.1) time units.
- *Stop-gradient at the solver boundary*: the upsampling model receives no gradient signal from the relaxed output, requiring a separate loss on its direct output (effectively phase 1 only). Rejected because it gives up on joint training entirely.
- *Two-phase with baked-in phase switching inside `TrainingSession`*: more monolithic; harder to skip phase 2 or checkpoint between phases. Rejected in favour of two separate `TrainingSession` objects constructed sequentially in `train_ctrl.jl`.

**Consequence**: the relaxation duration `T_relax` and timestep `dt_relax` must be short enough that gradients are numerically useful (Lyapunov instabilities grow exponentially past O(0.1) time units). The `KfRhs` used for relaxation is constructed from the same `Re`, `N`, and forcing wavenumber `n` as the data-generating solver — relaxing under different physics would push fields toward the wrong attractor.

## Phase-2 gradient compilation strategy

`loss_fn_relaxation` runs a Julia `for` loop over `n_steps` KF solver steps. When `Training.single_train_step!(AutoEnzyme(), ...)` traces this, Reactant fully unrolls the loop into the MLIR graph. For large `n_steps` (e.g., `T_relax=2.0`, `dt_relax=0.01` → 200 steps × 5 RK stages) the MLIR graph contains thousands of element-wise ops (one `*_broadcast_scalar` per complex multiplication per unrolled step), causing compilation to hang or fail with a naming-uniqueness error.

**Decision**: annotate the outer `for` loop in `relax()` with `Reactant.@trace`. Inside a `@compile` context, `@trace` promotes the loop bounds to traced values and lowers the loop to a single `stablehlo.while` op instead of unrolling — the entire 200-step rollout becomes one native XLA loop, which XLA differentiates at the HLO level. Outside `@compile` (CPU tests, plain Julia calls), `@trace` is a no-op and the loop runs normally. `Training.single_train_step!` continues to be used unchanged; the fix is entirely inside `relax()`.

**Alternatives considered**: pre-compiling the full loss+gradient in a separate `Reactant.@compile` call with `Enzyme.autodiff` inside, bypassing `Training.single_train_step!`. Rejected because calling `Enzyme.autodiff` inside a `@compile` context triggers nested MLIR function generation that replicates the same naming-uniqueness conflict as loop unrolling.

**Future**: when a neural operator replaces the KF solver as the fine-tuner, `freeze_upsampler = true` becomes meaningful (train only the operator). Full backprop through the operator is expected to be stable since it lacks chaotic Lyapunov growth.
