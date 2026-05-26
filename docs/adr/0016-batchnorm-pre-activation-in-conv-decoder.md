# BatchNorm + pre-activation replaces InstanceNorm + post-activation in ConvDecoder

## Status
Accepted

## Context

`ConvDecoder` used `InstanceNorm` throughout — in the `UpsampleBlock` DenseNet dense
layers, in the `conv_transpose` upsample path, and `LayerNorm` in the FC (`_build_mlp`)
layers. Activations were applied post-norm (after the convolution output).

Three problems motivated a change:

1. **InstanceNorm normalises per-sample per-channel**, computing statistics over the
   spatial dimensions `(H, W)` of a single feature map. At the small spatial sizes early
   in the ConvDecoder (e.g. 8×8 after the Fourier base), those statistics are estimated
   from only 64 values — noisy and unreliable.
2. **Post-activation in the DenseNet dense block** means each norm sees the output of a
   single conv in isolation, before concatenation. The running statistics therefore never
   reflect the full concatenated feature distribution that subsequent convolutions actually
   operate on.
3. **LayerNorm in the FC path** normalises over the feature dimension within each sample,
   which is inconsistent with how the downstream conv layers are normalised.

## Decision

Switch to **BatchNorm** and **pre-activation** throughout `ConvDecoder` and `_build_mlp`:

### DenseNet dense block (inside `UpsampleBlock`)

- **First dense conv** (`i=1`): no pre-activation. Receives `x_up` directly — there is no
  preceding BN output to normalize.
- **Subsequent dense convs** (`i > 1`): `BN(i·C_in) → act → SpectralCircConv(i·C_in → C_in)`.
  BN sees the full concatenated tensor at each step; statistics are computed after
  concatenation.
- **Projection conv**: `BN(n_convs·C_in) → act → Conv1×1(n_convs·C_in → C_out)`. Same
  pre-activation applied to the final concatenated tensor.
- `dense_norms` are non-uniform: `BatchNorm(i·C_in)` for the i-th norm (i = 2..n_dense),
  since each operates on a different-width concatenated input.
- A new `proj_norm::PN` field (`BatchNorm(n_convs·C_in)`) is added to `UpsampleBlock`.
  For `n_convs=1` (no dense layers), `proj_norm = BatchNorm(C_in)`.

### Conv-transpose upsample path (`upsample_mode = :conv_transpose`)

Post-activation is kept: `ConvTranspose → BN(C_in) → act`. Pre-activation does not apply
here — there is no concatenation and the input has not been through any conv in this block.

### FC layers (`_build_mlp`)

`Linear → BN → act` replaces `Dense(act) → LayerNorm`. Activation is removed from inside
`Dense`; `BatchNorm` is inserted before it. Consistent with the `BN → act` order chosen
for the conv layers.

### Tail convolutions

Unchanged: `tail_conv1 → act → tail_conv2`, no norms. The tail collapses channels to the
scalar stream function ψ; normalising a 1-channel intermediate would constrain the
amplitude of ψ and fight velocity-scale learning.

### Inference mode

`BatchNorm` accumulates running statistics during training. All eval paths
(`eval_decoder_vel`, `eval_decoder_vort`, diagnostic plots) must use
`Lux.testmode(st)` to switch to running-statistics mode. Training paths continue to use
`Lux.trainmode(st)` (or the default training state from `Lux.setup`).

## Alternatives considered

- **Keep InstanceNorm, switch to pre-activation**: retains the per-sample normalisation
  problem at small spatial sizes. Rejected.
- **GroupNorm**: more flexible than InstanceNorm but still per-sample; does not share
  statistics across the batch. Rejected in favour of BatchNorm's batch-level statistics.
- **No normalisation**: simpler, but empirically unstable for deep conv decoders with
  many DenseNet layers. Rejected.

## Consequences

- `UpsampleBlock` gains a `proj_norm` field; `dense_norms` have non-uniform channel
  counts. Existing checkpoints are incompatible (norm layer shapes differ).
- Eval and sampling code must explicitly call `Lux.testmode(st)` — forgetting this with
  a batch size of 1 produces degenerate (near-constant) outputs.
- BatchNorm with small batch sizes (< ~16) produces noisier statistics than InstanceNorm.
  The training `batch_size` (default 400) is well above this threshold.
