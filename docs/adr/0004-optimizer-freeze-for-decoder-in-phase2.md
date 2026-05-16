# ADR 0004: Optimizer Freeze (Not Stop-Gradient) for Decoder in Phase-2 FNO-Only Training

## Status
Accepted

## Context
Phase-2 trains the combined upsampling model + neural operator end-to-end via SWD. A common experiment is to hold the upsampling model fixed and train only the FNO — testing whether the neural operator alone can improve field quality on top of a frozen prior.

Two mechanisms can freeze the upsampler:

1. **Stop-gradient**: run the upsampler in inference mode; detach its output from the AD graph before the FNO. Enzyme never differentiates through the upsampler. Requires modifying `eval_decoder_vort(::UpsamplerWithFNO, ...)` or branching the loss function to detach the intermediate vorticity field.

2. **Optimizer freeze**: Enzyme still computes the full gradient (including through the upsampler), but `Optimisers.freeze!` marks the upsampler's optimizer `Leaf` nodes so parameter updates are never applied to them.

## Decision
Use optimizer freeze. After `session2 = TrainingSession(...)` is built, call:

```julia
Optimisers.freeze!(session2.tstate.optimizer_state.upsampler)
```

The `tstate.optimizer_state` tree mirrors `ps = (upsampler=..., fno=...)`. `Optimisers.freeze!` mutates the `Leaf` structs in the upsampler subtree in-place, setting `frozen = true` on each leaf. Subsequent calls to `single_train_step!` compute the full gradient but skip optimizer updates for frozen leaves.

## Alternatives considered
- **Stop-gradient**: Would shed the memory cost of storing upsampler intermediate activations during the backward pass, but requires surgical changes to the forward path or loss function, and the activation memory saving is negligible relative to the FNO's `rfft`/`irfft` intermediates which dominate phase-2 memory.

## Consequences
- Enzyme still differentiates through the upsampler; its intermediate activations are held in memory during the backward pass. This is acceptable because the upsampler is smaller than the FNO.
- `freeze_decoder_phase2` is recorded in `config.json` at checkpoint time so runs can be distinguished retrospectively.
- The `ReduceOnPlateau` scheduler tracks the overall SWD loss regardless of which parameters are frozen; no scheduler changes are needed.
- Unfreezing is trivially done with `Optimisers.thaw!` on the same subtree if needed in future experiments.
