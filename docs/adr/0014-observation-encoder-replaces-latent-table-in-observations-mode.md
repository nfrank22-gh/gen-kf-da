# ADR 0014 — Observation Encoder Replaces Per-Snapshot Latent Table in ObservationsMode

## Status
Accepted

## Context

In `ObservationsMode`, the **Latent Posterior** parameters `(mu, log_sigma)` were previously stored as two `latent_dim × n_train` parameter matrices and optimized directly by Adam alongside the upsampling model weights. This is the standard approach when there is no structure to exploit across snapshots.

Velocity observations at sensor locations carry direct information about the flow state. Rather than treating each snapshot's posterior independently, an **amortized** approach — a feedforward encoder that maps observations to `(mu, log_sigma)` — can share statistical strength across snapshots and generalize to unseen observations at inference time (one forward pass instead of an optimization loop per snapshot).

## Decision

Replace the per-snapshot latent parameter tables with an **Observation Encoder** (DeepSets architecture) in `ObservationsMode`. The encoder and upsampling model are jointly wrapped in an `ObservationEncoderDecoder` Lux container and trained end-to-end.

Key design choices:
- **DeepSets**: each sensor tuple `(u_i, v_i, sin(x_i), cos(x_i), sin(y_i), cos(y_i))` is embedded by a shared sub-MLP; mean pooling aggregates across sensors; a head MLP outputs `mu` and `log_sigma`. Mean pooling (not sum) keeps the aggregated embedding magnitude invariant to `n_meas`.
- **Fourier position features**: `(sin(x_i), cos(x_i), sin(y_i), cos(y_i))` encode the 2π-periodic torus topology so spatially nearby sensors (including across the periodic boundary) appear close to the encoder.
- **Features computed externally**: Fourier features are precomputed in the mode closures (once for fixed sensors, per batch for random sensors) and passed as a `(6, n_meas, batch_size)` tensor to the encoder. The `ObservationEncoderDecoder` struct stores no location data.
- **Scope**: fixed and random sensor variants of `ObservationsMode` only. Full-velocity mode is excluded — the encoder would receive the entire `2·N²`-dim field, a qualitatively different problem.

`VorticityMode` is unchanged — it retains the per-snapshot latent parameter tables and the **Latent LR Multiplier**.

## Consequences

- `latent_mu` / `latent_log_sigma` are not added to `ps` when the model is an `ObservationEncoderDecoder`; `latent_lr_multiplier` has no effect in that case.
- The loss functions for `ObservationsMode` no longer receive `cols` for latent lookup; `mu` and `log_sigma` come from the encoder forward pass on the batch's observations.
- At eval time, unconditional generation still samples `z ~ N(0,I)` and calls the decoder directly — the encoder is bypassed.
- `build_model_from_config` and `config.json` must be extended with encoder hyperparameters (`encoder_hidden`, `encoder_head_hidden`).
