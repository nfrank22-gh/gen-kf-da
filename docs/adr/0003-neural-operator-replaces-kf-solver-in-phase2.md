# ADR 0003: Neural Operator Replaces KF Solver in Phase-2 Physics Refinement

## Status
Accepted

## Context
Phase-2 training originally used the KF solver as the physics refinement step (see ADR 0002). This required a hand-coded adjoint (`relax_and_store` / `relax_adj_full` / `_kf_step_batched_adjoint`) because Enzyme cannot trace through the IMEX RK solver internals. The adjoint added ~200 lines of bespoke backward code and a non-standard 4-step training loop (forward on GPU → SWD on CPU → solver adjoint on GPU → VJP update).

## Decision
Replace the KF solver in phase 2 with an FNO-based **neural operator** that maps a vorticity field to a refined vorticity field. The operator is applied autoregressively `n_fno_steps` times. It is trained from scratch during phase 2, jointly with the upsampling model, via a single end-to-end SWD loss. No pre-training on KF solver trajectories is performed.

The combined upsampling model + neural operator is wrapped in a single Lux container so `Training.single_train_step!` updates both sets of parameters in one call — the same pattern as phase 1.

The KF solver adjoint code (`relaxation.jl`) is deleted entirely.

## Alternatives considered
- **Pre-train the neural operator separately, then freeze during phase 2**: Simpler phase-2 loop, but requires a separate supervised training pipeline against KF solver rollouts and adds a hyperparameter for how long to pre-train. The FNO only needs to learn to be an attractor-projecting map, not a precise dynamics simulator, so pre-training is unnecessary overhead.
- **Keep the KF solver, wrap it in a custom Lux layer**: Would allow the same single-`tstate` training loop, but Enzyme still cannot trace through the solver, so the hand-coded adjoint would remain.
- **Use the KF solver for inference only, neural operator for training**: Incoherent — inference and training would use different physics refinement dynamics, making phase-2 training misleading.

## Consequences
- The hand-coded solver adjoint (~200 lines) is deleted; phase-2 training collapses to the same structure as phase 1.
- Gradients flow end-to-end through both stages via AutoEnzyme on Reactant arrays.
- The neural operator has no guaranteed physics consistency (unlike the KF solver); it learns to project toward the attractor purely from the SWD signal.
- New hyperparameters in `train_ctrl.jl`: `n_fno_steps`, `n_modes`, `n_fno_layers`, `fno_channels`.
- Eval SWD in phase 2 runs the full pipeline (upsampling model → neural operator) rather than the upsampling model alone.
