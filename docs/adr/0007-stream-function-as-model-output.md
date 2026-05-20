# ADR 0007 — Stream function as native model output

**Status:** Accepted

## Context

All model architectures (`VelFourierDecoder`, `ConvDecoder`, `FourierNeuralOperator`) produced
divergence-free velocity `(u, v)` as their native output, enforcing incompressibility via the
**Incompressibility Projection** (Leray projector) applied in spectral space as a hard architectural
constraint. While this guaranteed ∇·u = 0, it had two drawbacks:

1. **Redundant constraint**: the Leray projector projects onto the solenoidal subspace, but the
   network is already free to learn any vector field — the projection discards the irrotational
   component that the network chose to produce, which may impede learning.
2. **Higher-dimensional output**: each architecture output two channels `(u, v)`, requiring 2×
   the spectral DOFs (`VelFourierDecoder`) or 2-channel final convolutions (`ConvDecoder`, FNO).

The velocity field of a 2D incompressible flow is fully determined by a single scalar — the
stream function `ψ` — via `u = ∂ψ/∂y`, `v = -∂ψ/∂x`. A model that outputs `ψ` directly cannot
produce a non-solenoidal field regardless of its weights.

## Decision

All model architectures now output a scalar **stream function** `ψ` as their native representation.
Velocity and vorticity are derived analytically in spectral space:

```
û  =  iky · ψ̂
v̂  = −ikx · ψ̂
ω̂  =  |k|² · ψ̂
```

The DC mode `ψ̂(0,0)` is zeroed explicitly: it has no effect on `u`, `v`, or `ω` (since
`iky·0 = 0` and `ikx·0 = 0`), so leaving it free wastes a learnable DOF. This also preserves
consistency with the KF solver's DC-mode handling in `SpectralGrid`.

Architecture changes:

- **`StreamFourierDecoder`** (renamed from `VelFourierDecoder`): MLP outputs free `ψ̂`
  coefficients — 2× real values (re, im) per mode instead of 4×. DC mode zeroed. Private
  primitive `_decode_psi_hat` replaces `_decode_vel_hat`.
- **`ConvDecoder`**: final conv reduced from 2 output channels to 1 (`ψ`). Incompressibility
  Projection removed. `(u, v)` derived from `ψ̂` after rfft.
- **`FourierNeuralOperator`**: lift changed from 2→C to 1→C, project changed from C→2 to C→1,
  operating on `ψ` directly. Per-step Leray projections inside `eval_decoder_vel` are removed;
  divergence-freeness is maintained throughout autoregressive refinement by construction.

The public interface (`eval_decoder_vel`, `eval_decoder_vort`) is unchanged at call sites.
Both are derived from `ψ̂` inside each architecture's implementation.

## Alternatives considered

**Keep velocity output with Leray projection** (prior approach): explicit hard constraint,
familiar to fluid-dynamics practitioners. Rejected because incompressibility is already
structural in `ψ`-space — the projection is architectural overhead with no learning benefit,
and the output dimensionality is unnecessarily doubled.

**Output vorticity `ω` and invert to velocity**: the 1/|k|² spectral filter amplifies
high-frequency vorticity noise into velocity, producing over-smooth velocity fields. Ruled out
on prior experience (see history of VortFourierDecoder).

**FNO operates on `(u, v)` while upsampler outputs `ψ`**: inconsistent representation across
pipeline stages; requires a projection step at the upsampler/FNO boundary and re-introduces
per-step Leray projections inside the FNO loop. Rejected in favour of uniform `ψ` throughout.

## Consequences

- Incompressibility is a structural guarantee — not a constraint applied after the fact — for
  both the upsampling model and all autoregressive FNO refinement steps.
- `StreamFourierDecoder` MLP output dimension halves relative to `VelFourierDecoder`.
- `ConvDecoder` final conv goes from 2 channels to 1.
- FNO lift/project channel counts change from 2 to 1; the `SpectralGrid` stored in
  `UpsamplerWithFNO` is still used for `(u, v)` derivation at eval time.
- Existing phase-1 and phase-2 checkpoints are **incompatible** with the new architecture.
- `eval_decoder_vel`, `eval_decoder_vort`, `VorticityMode`, and all plot helpers are
  call-site-compatible — they receive `(u, v)` or `ω` derived from `ψ` without changes at
  their call sites.
