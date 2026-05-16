# ADR 0006: Variational Latent Posterior Replaces Point Estimates

**Status**: Accepted

## Context

Previously, each training snapshot was assigned a point-estimate latent vector stored as a column of `ps.latents` (`latent_dim × n_train`). KL regularization was a simplified L2 penalty: `kl_weight * 0.5 * mean(sum(x.^2, dims=1))`, which penalizes drift from zero but does not constrain the variance of the learned representations.

## Decision

Replace the point-estimate latent matrix with a per-snapshot variational posterior N(mu, sigma²), stored as two parameter matrices:

- `ps.latent_mu` (`latent_dim × n_train`): initialized from N(0,I)
- `ps.latent_log_sigma` (`latent_dim × n_train`): initialized to 0 (sigma = 1)

During training, latent vectors are drawn via the reparameterization trick:

```julia
z = ps.latent_mu[:, cols] .+ exp.(ps.latent_log_sigma[:, cols]) .* eps
```

where `eps ~ N(0,I)` is generated in the training loop, converted to a Reactant array, and passed explicitly into the loss function alongside `thetas` and `cols`. This is required because random number generation cannot occur inside a Reactant-compiled trace.

KL regularization is replaced with the exact KL from N(mu, sigma²) to N(0,I):

```julia
sigma = exp.(log_sigma)
kl = 0.5f0 * mean(sum(sigma.^2 .+ mu.^2 .- 1f0 .- 2f0 .* log_sigma, dims=1))
```

At eval time, latent vectors are sampled directly from N(0,I); the learned posteriors are not used.

When `fix_latents_phase2 = true`, both `ps.latent_mu` and `ps.latent_log_sigma` are frozen together via `Optimisers.freeze!`.

## Alternatives Considered

- **Keep point estimates**: Simpler, but L2 regularization only controls the mean of the latent distribution, not its variance. The posterior spread is unconstrained.
- **Shared sigma across snapshots**: Reduces parameters but prevents the model from learning per-snapshot uncertainty. Rejected.

## Consequences

- Checkpoints from before this change are incompatible (`ps.latents` key no longer exists).
- The loss function signatures gain an `eps` argument.
- `kl_weight` retains its role as the beta scalar (beta-VAE interpretation).
