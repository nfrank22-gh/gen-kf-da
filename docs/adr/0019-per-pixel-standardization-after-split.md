# ADR 0019 — Per-pixel z-score standardization applied after train/eval split

## Status
Accepted

## Context
The **Mahalanobis Projection** requires PCA of the training feature space. PCA on uncentered data conflates the mean vorticity field with the principal modes of variation, so the data must have mean zero before SVD. Rather than centering inside the PCA routine, we add a general-purpose **Snapshot Standardization** step so the model trains and evaluates in a space that already has mean 0 and unit per-pixel variance.

Two ordering options were considered:
- **Standardize before split** — simple call site, but mean/std are contaminated by eval snapshots (data leakage).
- **Standardize after split** — stats computed from training snapshots only, then applied to eval snapshots using the same statistics.

## Decision
Standardize after split. `standardize_snapshots(train_snaps)` computes per-pixel `μ[i,j]` and `σ[i,j]` from training data and returns the standardized snapshots plus both stat fields. `apply_standardization(eval_snaps, μ, σ)` applies the pre-computed stats to eval data. The model trains and outputs vorticity in standardized space; diagnostic plots un-standardize (`ω_phys = ω_std .* σ .+ μ`) before display.

## Consequences
- Old checkpoints trained without standardization are not directly comparable to new runs in the same output space.
- `mean_field` and `std_field` (flattened `N×N` Float32 arrays) are saved in `config.json` so a checkpoint can be reproduced without the original trajectory.
- PCA for the Mahalanobis Projection is computed on already-standardized training data; no separate centering step is needed inside the PCA routine.
- Eval SWD compares standardized generated vorticity against standardized eval vorticity — both in the same space, so the metric remains valid.
