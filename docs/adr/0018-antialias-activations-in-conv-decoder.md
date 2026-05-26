# ADR 0018 — Anti-aliased activations in ConvDecoder

**Status:** Accepted

## Context

Elementwise nonlinearities applied to spatially-bandlimited feature maps generate frequency content above the current Nyquist frequency. In a periodically-padded convolutional network operating at a fixed spatial resolution, those high-frequency components fold back (alias) into the signal rather than being discarded. This aliasing is a plausible source of high-frequency ringing artifacts in the generated stream function `ψ`.

## Decision

Every 4D spatial activation in `ConvDecoder` is replaced by the anti-aliased form:

```
spectral_upsample_2x → act → spectral_downsample_2x
```

implemented as `antialias_act(act, x)` in `conv_decoder.jl`. The sites are:

| Location | Activation |
|---|---|
| `UpsampleBlock` conv_transpose branch | `act` |
| `UpsampleBlock` dense loop (i ≥ 2) | `act` |
| `UpsampleBlock` post-`proj_norm` | `act` |
| `UpsampleBlock` post-`proj_out_norm` | `act` |
| `GEBlock` bottleneck | `act` |
| `GEBlock` gating | `sigmoid` |
| `_eval_psi_physical` tail | `act` |

FC/MLP activations inside `_build_mlp` are **not** wrapped — they operate on 1D latent vectors where 2D spatial aliasing does not apply.

`spectral_downsample_2x` is the exact spectral inverse of `spectral_upsample_2x`: rfft → keep the low-frequency corner → irfft → scale by `1/4`. The `1/4` factor ensures `spectral_downsample_2x(spectral_upsample_2x(x)) == x` for any bandlimited `x`. The low-pass filter removes the aliasing products introduced by the nonlinearity before returning to the original resolution.

## Alternatives considered

**Bilinear upsample/downsample**: cheaper but uses a non-ideal filter with spectral leakage, and inconsistent with the spectral idioms already in the codebase.

**No anti-aliasing**: simpler, but leaves aliasing products from activations in the signal.

**Apply only to a subset of activation sites**: partial fix that leaves some aliasing paths open; complexity without full benefit.

## Consequences

- Every 4D activation now costs two additional FFT/IFFT pairs (one upsample, one downsample). This roughly doubles the per-activation compute at each spatial resolution.
- The `spectral_upsample_2x` / `spectral_downsample_2x` pair is Reactant/XLA-compatible (no in-place ops, no FFTW-specific paths), so training on GPU is unaffected.
- Round-trip fidelity (`spectral_downsample_2x ∘ spectral_upsample_2x ≈ id`) is verified by the `spectral_downsample_2x round-trip` test in `test_conv_decoder.jl`.
