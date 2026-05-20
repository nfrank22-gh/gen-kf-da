# ADR 0013 — Symmetry Reduction Replaces Learned Periodic Translation Parameters

**Status**: Accepted

## Context

2D Kolmogorov flow with forcing wavenumber n has a 16-element discrete symmetry group (8-fold shift-reflect S × 2-fold rotation R) plus a continuous streamwise translation symmetry T_s. Training snapshots drawn from a long trajectory are spread across all 16 discrete equivalence classes and all continuous translations, forcing the model to either learn all copies or waste capacity representing the same physical state multiple times.

The previous approach (**ADR 0008**) handled the continuous translation by adding learned per-snapshot periodic translation parameters (`shift_mu`, `shift_log_sigma`) to `ps`, sampled via the reparameterization trick, and regularised with a half-KL penalty. The discrete symmetries were left unaddressed.

## Decision

Remove the learned shift parameters entirely. Instead, apply **symmetry reduction** as a one-time preprocessing step (`reduce_trajectory`) immediately after loading the trajectory in `train_ctrl.jl`, before the train/eval split.

The procedure follows Cleary & Page (Phys. Rev. E 112, 055105, 2025), Section II.2:

1. **Shift-reflect (discrete)**: identify the sector of arg(ω̂(0,1)) ∈ [0, 2π) using `sector = floor(θ / (π/n))`. Apply correction S^k in physical space (y-shift + optional x-flip + optional negate) where `k = mod((n−1)·sector, 2n)`. For n=4: k = 3·sector mod 8. This matches Table I of the paper exactly.

2. **Rotation (discrete)**: if imag(ω̂(0,n)) < 0, apply R = complex conjugation of ω̂ (equivalent to spatial rotation by π).

3. **Continuous translation (method of slices)**: align the (kx=1, ky=0) Fourier mode to the positive real axis by multiplying ω̂ by exp(−i·kx·ϕ) where ϕ = arg(ω̂(1,0)).

Codebase spectral convention: ω̂[ky_idx, kx_idx], i.e. dim 1 = ky, dim 2 = kx.

## Consequences

- **Simpler model**: `_apply_periodic_shift`, `shift_mu`, `shift_log_sigma`, `_shift_kl`, and all `shifts` arguments are removed from the model, loss functions, and training loop.
- **Smaller parameter count**: 2·n_train shift parameters removed from `ps`.
- **Richer latent space**: the model learns one representative per physical state rather than spreading probability mass across 16 symmetry copies. Consistent with Cleary & Page's finding that explicit symmetry reduction yields a richer, more interpretable latent space.
- **`snapshot_DA.jl` needs updating**: that script still constructs shift parameters for the DA loop and must be rewritten to operate in the reduced space.
- **Eval snapshots are also reduced**: `reduce_trajectory` is applied before `split_trajectory`, so eval SWD compares generated samples and held-out snapshots in the same reduced coordinate system.

## Alternatives considered

- **Data augmentation** (randomly applying symmetries each epoch): does not guarantee equivariance and was found to give higher reconstruction error than explicit reduction (Cleary & Page, Fig. 5).
- **Keep learned shifts**: simpler but leaves discrete symmetries unaddressed and wastes model capacity.
