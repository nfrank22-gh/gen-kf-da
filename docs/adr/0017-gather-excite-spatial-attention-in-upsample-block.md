# ADR 0017 — GE-θ+ spatial channel attention in UpsampleBlock

**Status:** Accepted

## Context

`UpsampleBlock` uses `SpectralCircConv` in every dense-block conv and in both tail convs (ADR 0015). The spectral branch of `SpectralCircConv` provides global structure by mixing channels across the lowest `k_max` Fourier modes in frequency space. What it does not do is gate individual channels based on their own spatial activation pattern — it mixes channels globally across modes, but the gating is learned per-mode, not per-spatial-context.

Gather-Excite (GE-θ+) addresses a complementary axis of attention: for each spatial location, gather context over a local neighbourhood via a depth-wise convolution, then use a bottleneck FC to produce a per-channel sigmoid gate that multiplicatively rescales the feature map. The two mechanisms are orthogonal:

- `SpectralCircConv` spectral branch: mixes channels across Fourier modes (global, frequency-domain)
- GE-θ+: gates channels by their local spatial activation pattern (spatial-domain, per-channel)

## Decision

Add an optional `GEBlock` sublayer to `UpsampleBlock`, inserted between the pre-activation projection normalisation and the 1×1 projection convolution — no additional norms or activations between `GEBlock` output and `proj`. Disabled by default (`use_ge=false`) so the change is backward-compatible.

**Forward pass position:**
```
act(proj_norm(h))  →  GEBlock  →  proj (Conv 1×1)
```

**GEBlock internals:**

1. **Gather**: depth-wise `CircConv(C, C, θ; groups=C)` — one filter per input channel, circular padding preserved, kernel size θ set per-block via `ge_kernel_sizes::Vector{Int}` on `ConvDecoder`.
2. **Excite**: `Conv1×1(C → max(C÷r, 4)) → act → Conv1×1(max(C÷r, 4) → C) → sigmoid`, where `r` is `ge_reduction::Int` (default 4) and `act` is the block's activation function.
3. **Gate**: `x * (1 + sigmoid(context))` — scale ∈ [1,2], amplification-only residual.

**When `use_ge=false`**: `ge = NoOpLayer()` — zero parameters, identity forward pass, no overhead.

## Alternatives considered

- **Global GAP gather (SE-style)**: collapses the spatial dimension to a single per-channel scalar, losing the spatial attention that depth-wise conv provides. Rejected; if only channel attention were needed, `SpectralCircConv`'s spectral branch already provides cross-channel mixing more expressively.
- **GE without FC bottleneck (direct sigmoid on depth-wise output)**: simpler, no cross-channel mixing in the excite step. Could be added as a variant later if the bottleneck proves unnecessary.
- **Placement before `proj_norm`**: `BatchNorm` would normalise away the attention scaling immediately after GE computes it. Rejected.
- **Scale ∈ [0,1] (plain sigmoid, no +1 residual)**: allows complete channel suppression. Rejected; the `+1` residual ensures the identity path is preserved when the gate is near zero, which stabilises early training.
- **Fixed ReLU between bottleneck FC layers**: the paper uses ReLU here, but using the block's `act` is consistent with the rest of `ConvDecoder` (single activation function threaded through) and avoids a new hard-coded choice.

## Consequences

- `UpsampleBlock` gains a `ge::GE` type parameter; `:ge` is added to the `AbstractLuxContainerLayer` field tuple between `:proj_norm` and `:proj`.
- `CircConv` constructor gains a `groups::Int=1` keyword argument (no change to existing call sites).
- **Backward compatibility**: `use_ge=false` (default) → `ge = NoOpLayer()` with no parameters; existing checkpoint `ps`/`st` layouts are unaffected for this default.
- **`use_ge=true`**: existing checkpoints are incompatible — `ps.blocks.block_i` now includes `ge` sublayer fields.
- Forward-pass cost when `use_ge=true`: one depth-wise `CircConv` + two 1×1 `Conv` calls per `UpsampleBlock`.
