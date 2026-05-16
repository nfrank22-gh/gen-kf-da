# ADR 0005: Optimized Latent Matrix Replaces Random Sampling

## Status
Accepted

## Context
Phase-1 training previously drew fresh random latent vectors from N(0,I) each epoch (or cached them if `fix_x=true`). These vectors were passed as a separate argument `x` to the loss function and were never optimized — the model had to learn to map arbitrary Gaussian noise to plausible vorticity fields.

## Decision
Replace per-epoch random sampling with a `latent_dim × n_train` matrix of point-estimate latent vectors (`ps.latents`) that are jointly optimized with the upsampling model weights via Adam.

Key consequences of this choice:

- **Latents in `ps`**: The latent matrix is merged into the model parameter NamedTuple so that `Training.single_train_step!` computes gradients and applies optimizer updates to it automatically.
- **Loss indexes `ps.latents[:, cols]`**: The loss function extracts the batch of latent vectors internally using `cols` rather than receiving `x` as a separate argument. This is the only way to route gradients back through `ps`.
- **KL regularization**: A fixed-weight L2 penalty (`kl_weight * 0.5 * mean(sum(x.^2, dims=1))`) is added to the SWD loss to prevent the latent vectors from drifting arbitrarily far from N(0,I). This is analytically equivalent to the KL divergence between point-mass Gaussians at each `z_i` and the standard normal.
- **Eval samples from N(0,I)**: The eval SWD draws fresh latent vectors from the prior, not from `ps.latents`, so it measures generalization rather than memorization.
- **Phase-2 behavior**: A `fix_latents_phase2` bool controls whether `ps.latents` is frozen (via `Optimisers.freeze!`) or continues to be optimized alongside the neural operator in phase 2.

## Alternatives Considered

**VAE-style (mu + log-variance per snapshot)**: Doubles the latent parameter count, requires reparameterization sampling, and adds a more complex KL term. Rejected because the SWD loss is already distributional; the simpler point-estimate approach achieves the same regularization goal.

**Separate optimizer for the latent matrix**: Would allow per-LR control and avoid Adam momentum drift on un-touched columns. Rejected because with `n_full ≈ 2`, each latent is touched every other step; momentum decay between updates (`beta2^1 ≈ 0.999`) is negligible, and keeping everything in one optimizer preserves the existing `single_train_step!` loop.

**Keep `x` as external argument, use custom gradient step**: Preserves the current loss signature but requires bypassing `Training.single_train_step!` entirely. Rejected as higher complexity for no meaningful benefit.

## Trade-offs
The latent matrix grows with the training set (`latent_dim × n_train` float32 values). For `latent_dim=100, n_train=800` this is 320 KB — negligible. The loss function signature changes (adds `cols` and `kl_weight`, drops `x`), and `BatchSampler` loses the `fix_x` / `_x_cache` logic that is now obsolete.
