# ADR 0008: Variational Periodic Translation Parameters

**Status:** Accepted

## Context

The KF solver operates on a 2D periodic torus `[0, 2π] × [0, 2π]`. Solutions are statistically translation-invariant. Convolutional networks are also translation-equivariant, so the `ConvDecoder`'s output has no preferred origin — the conv backbone wastes capacity deciding where to "place" the flow pattern rather than learning its shape. The `StreamFourierDecoder` has the same issue.

## Decision

Add per-snapshot **Periodic Translation Parameters** to the outer `ps`: two `2 × n_train` matrices `ps.shift_mu` and `ps.shift_log_sigma`. At each training step, shifts are sampled via the reparameterization trick:

```
shift = shift_mu[:, cols] + exp(shift_log_sigma[:, cols]) * eps_shift,  eps_shift ~ N(0,I)
```

The shift is applied as a spectral phase multiplication on `ψ̂`:

```
ψ̂_shifted = ψ̂ · exp(-i·(kx·dx + ky·dy))
```

which rigidly translates the physical-space stream function on the torus. `eval_decoder_vel` and `eval_decoder_vort` gain a `shifts` argument (a `2 × B` matrix between `x` and `ps`). Applied to both `StreamFourierDecoder` and `ConvDecoder`.

**Initialization:** `shift_mu ~ Uniform(-π, π)`, `shift_log_sigma = log(shift_sigma_target)` so sigma starts at the target.

**Regularization (Shift KL Regularization):** KL from N(0, sigma²) to N(0, sigma_target²), summed over x and y, averaged over batch, weighted by `kl_weight * 2 / latent_dim`:

```
shift_kl = mean(sum(log(sigma_t/sigma) + sigma²/(2·sigma_t²) - 0.5, over {x,y}))
```

No regularization on `shift_mu` — any translation is equally valid on the periodic torus.

**`shift_sigma_target`:** hyperparameter (default 0.25 radians, ≈4% of domain), controls how tightly sigma is pulled toward a committed per-snapshot shift. Shared weight `kl_weight * 2/latent_dim` keeps per-dimension contribution equal to the latent KL.

**Eval / Snapshot DA:** shifts sampled from Uniform(0, 2π) to cover the full torus, consistent with the translation symmetry of the KF system.

## Alternatives considered

**Point estimates (deterministic shifts):** Simpler, but provides no gradient smoothing from stochastic sampling and no mechanism to control shift uncertainty.

**No regularization on sigma:** Sigma could grow unboundedly, making shifts uniformly random at every training step — equivalent to having no shift at all and forcing the decoder to handle all translations again.

**Separate `shift_kl_weight`:** Rejected because `shift_sigma_target` already controls regularization tightness; a second weight would be redundant.

**Regularize toward sigma=0 (L2 on sigma):** Would collapse the distribution entirely; no principled floor.

**Uniform(0, 2π) eval distribution vs. prior N(0, sigma_target²):** Uniform was chosen to ensure full torus coverage at eval, consistent with the statistical translation symmetry of the KF system, even though training concentrates near learned `shift_mu`.

## Consequences

- `ps` gains `shift_mu` and `shift_log_sigma` (replaces the earlier `latent_shifts` point-estimate matrix).
- All `eval_decoder_vel` and `eval_decoder_vort` callsites take a `shifts` argument.
- Loss functions gain `eps_shift` and `shift_sigma_target` arguments.
- `shift_sigma_target` is recorded in the checkpoint `config.json`.
