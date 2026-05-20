# ADR 0012 — Decoupled latent learning rate (10× multiplier)

## Status
Accepted

## Decision
All variational parameters — `latent_mu`, `latent_log_sigma`, `shift_mu`, `shift_log_sigma` — are trained at `lr * latent_lr_multiplier` (default 10×) rather than the global decoder learning rate. Applied by calling `Optimisers.adjust!` on those four subtrees of the optimizer state tree after `TrainingSession` builds `tstate`, and re-applied after every LR scheduler step to preserve the ratio as the global LR decays.

## Context
The latent matrix has `latent_dim × n_train` entries (e.g. 200 × 800). Each per-snapshot column receives gradient signal only when that snapshot appears in the current batch — far fewer effective updates per epoch than the shared decoder weights, which accumulate gradients from every batch. A higher LR compensates for this sparsity. The same argument applies to the 2 × n_train shift parameters.

Both LRs are printed each epoch so the ratio can be verified in training logs.

## Alternatives considered
- **Gradient scaling** — scale latent gradients before the optimizer step rather than adjusting the optimizer state; equivalent effect but less transparent and harder to compose with the existing LR scheduler.
- **Separate `TrainState` for latents** — clean conceptually but would require two `single_train_step!` calls per batch and a significant restructuring of `TrainingSession`.
- **No decoupling (current)** — all parameters share the same LR; latent codes adapt slowly and may not converge within the training budget.

## Consequences
- `latent_lr_multiplier` (default 10) is a new hyperparameter exposed in `train_ctrl.jl` and saved in `config.json`.
- Every call to `Optimisers.adjust!` (i.e. every LR scheduler step) must be followed by re-applying the multiplier to the four subtrees; this is encapsulated in a private helper `_adjust_latent_lr!` in `training_session.jl`.
- The global LR printed by the scheduler already matches what the decoder sees; the separate latent LR print makes the effective latent rate visible without arithmetic.
