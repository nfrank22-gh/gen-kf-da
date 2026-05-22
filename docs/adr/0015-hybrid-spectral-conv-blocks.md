# ADR 0015 — Hybrid local-global conv blocks in ConvDecoder (SpectralCircConv)

**Status:** Accepted

## Context

`ConvDecoder` uses circular-padded convolutions (`CircConv`) throughout its `UpsampleBlock` stages and tail. Circular convolutions have a kernel-bounded receptive field: a kernel of size `k` on a resolution-`H` map sees at most `k×k` spatial context, even after several stacked layers. For 2D Navier-Stokes flow, vorticity structures are globally coupled through pressure and incompressibility — a single vortex at one corner influences the entire domain. We wanted the conv backbone to capture both fine local texture and long-range global correlations within each layer, not just after many stacked operations.

## Decision

Replace every `CircConv` in `ConvDecoder` — the dense-block convolutions, the main convolution, and both tail convolutions (but **not** the 1×1 projection convs) — with `SpectralCircConv`, a hybrid layer that runs two parallel branches and **sums** their outputs:

1. **Conv branch** — the original `CircConv` with circular padding (unchanged; local receptive field).
2. **Spectral branch** — truncated FNO-style: `rfft` the input → apply a learned complex `(k_max, k_max, C_out, C_in)` channel-mixing weight matrix to the two low-frequency corner regions of the 2D rfft grid (`[1:k_max, 1:k_max]` and `[1:k_max, W-k_max+1:W]`) → zero-pad the rest → `irfft` back to physical space.

The two branches are summed (not concatenated) before normalization and activation, so `SpectralCircConv` is a drop-in replacement for `CircConv` with identical input/output shapes.

**Weight parameterization:** Complex weights are stored as four split Float32 tensors `(W_lo_re, W_lo_im, W_hi_re, W_hi_im)` to keep all Lux parameters plain `Float32`, consistent with the `complex.(re, im_)` pattern used elsewhere in the model.

**Configuration:** `k_max` is set per-block via `spectral_modes::Vector{Int}` (keyword arg, default `fill(4, n_blocks)`) and separately for the tail via `tail_spectral_modes::Int` (default `4`), mirroring the existing `kernel_sizes` / `tail_kernel` interface.

## Alternatives considered

- **Concatenation instead of sum** — doubles the output channel count, requiring downstream channel-count accounting changes throughout `UpsampleBlock`. Sum is a drop-in replacement with no ripple effect and is also how FNO combines its branches.
- **Full-spectrum FNO weights** — `(H÷2+1, W, C_out, C_in)` per layer. Parameter count scales with spatial resolution; at 64×64 this is large. Truncation to `k_max` modes is the standard FNO practice and keeps parameters controlled.
- **Single weight tensor (channel-mixing only, shared across frequencies)** — loses wavenumber-dependent expressiveness. The corner-block approach (two `k_max × k_max` weight matrices) is richer without excessive cost.

## Consequences

- **Parameter count increases** per `SpectralCircConv`: 4 × `k_max² × C_out × C_in` extra Float32 parameters (real + imag for two corners). With default `k_max=4` this is 64 × `C_out × C_in` extra params per layer, small relative to the conv weights.
- **Forward-pass cost increases** by one `rfft` + one `irfft` per `SpectralCircConv` call. Arrays are small (8×8 to 64×64) so this is acceptable.
- **Old checkpoints are incompatible** — `ps` now has `W_lo_re` etc. under each conv sublayer. `build_model_from_config` falls back to `spectral_modes=fill(4, n_blocks)` for checkpoints without the new keys, which reconstructs the model structure but weights cannot be loaded from an old checkpoint.
- **`CircConv` is retained** as a struct — it is used internally by `SpectralCircConv` as the conv branch sublayer.
