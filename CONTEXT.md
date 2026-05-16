# gen_DA_project

A Julia codebase that generates training data and trains a two-stage generative model to learn the distribution of 2D Kolmogorov-forced Navier-Stokes flow states, with the long-term goal of data assimilation from sparse observations.

## Language

**KF Solver**:
The pseudo-spectral Navier-Stokes solver driven by sinusoidal Kolmogorov forcing. Produces the ground-truth flow data that the generative model is trained on.
_Avoid_: simulator, forward model, DNS solver

**Trajectory** (short: traj):
A time sequence of vorticity snapshots produced by the KF solver. The primary output of data generation and the source of training data.
_Avoid_: rollout, flow snapshot sequence, ground-truth field

**Vorticity**:
The curl of the velocity field; the primary variable of the KF solver. Stored in spectral space as Fourier coefficients (`omega_hat`) and in physical space as a real array (`omega`). Velocity `(u, v)` is derived from vorticity via the stream function.
_Avoid_: flow state, flow field

**Observation**:
A sparse sample of the flow used to train or evaluate the generative model. Currently: velocity `(u, v)` at a fixed set of spatial grid locations sampled from a trajectory. Future: passive tracer particle positions.
_Avoid_: measurement, data point

**Sensor Array**:
The fixed set of spatial grid locations at which velocity observations are taken. Defined once before training and held constant for the entire training run. Represents a real physical sensor layout.
_Avoid_: measurement locations, observation points, sensor positions

**Training Horizon**:
The time cutoff `T_train` applied to a trajectory; only snapshots up to `T_train` are used for training. Snapshots after `T_train` form the held-out eval set.
_Avoid_: training split, train/test split

**Generative Model**:
The full two-stage pipeline that maps a latent vector to a physical-space vorticity field: an upsampling model followed by optional physics refinement. The SWD loss is always computed on the final output (after physics refinement if enabled).
_Avoid_: decoder, prior, surrogate model

**Upsampling Model**:
The first stage of the generative model. A neural network that maps a latent vector to an approximate vorticity field (N×N). Two architectures are available: `VortFourierDecoder` and `ConvDecoder`. The output may lie off the system's attractor; physics refinement corrects this.
_Avoid_: decoder, generative model (use that term for the full pipeline)

**VortFourierDecoder**:
The MLP-based upsampling model architecture. An MLP maps the latent vector to Fourier coefficients, which are spectrally padded to the target resolution and inverse-FFT'd to physical space.
_Avoid_: Fourier decoder, spectral decoder

**ConvDecoder**:
The convolutional upsampling model architecture. FC layers map the latent vector to a small spatial feature map, which is progressively upsampled to N×N via a series of spectral upsampling blocks, then collapsed to a 1-channel vorticity field by a final convolution.
_Avoid_: conv decoder, upsampling decoder

**Physics Refinement**:
The second stage of the generative model. A dynamical process that advances the upsampling model's output forward in time, projecting it toward the system's attractor. The first implementation is relaxation via the KF solver; future implementations may use neural operators.
_Avoid_: fine-tuning, post-processing, correction step

**Relaxation**:
The KF-solver-based implementation of physics refinement. The upsampling model output is converted to spectral space and integrated by the KF solver for `T_relax` physical time units (with timestep `dt_relax`), then converted back to physical space. The solver's dynamics decay transients introduced by the upsampling model and pull the field toward the attractor.
_Avoid_: solver rollout, fine-tuning, attractor projection

**Spectral Upsampling Block**:
One stage of the `ConvDecoder`. Doubles spatial resolution via spectral interpolation (FFT zero-pad → IFFT), applies `n_convs_per_block` circular-padded convolutions at constant channel width, adds the upsampled input as a skip connection, then reduces channels via a 1×1 convolution.
_Avoid_: upsampling layer, decoder block

**Latent Vector**:
The input to the generative model, drawn from (or optimized within) a Gaussian distribution. Denoted `x` in code.
_Avoid_: latent code, latent variable, noise vector

**Spin-up**:
An initial phase of KF solver integration (duration `T_spinup`) that is discarded to allow transients to decay before recording a trajectory.
_Avoid_: burn-in, equilibration, warm-up

**Sliced Wasserstein Distance (SWD)**:
The training and evaluation loss. Projects sample sets onto random unit directions and averages the 1-D Wasserstein distance. Always computed in physical space: either over flattened vorticity fields (`N×N`) or over velocity observations at sensor locations.
_Avoid_: Wasserstein loss, Earth mover's distance, OT loss

**Eval SWD**:
The physical-space sliced Wasserstein distance computed every `eval_every` epochs against held-out vorticity snapshots (flattened `N×N` arrays). Measures generalisation to unseen flow states without requiring observations.
_Avoid_: validation loss, test loss

**Checkpoint**:
The saved model artefacts after a training phase. Phase 1 (upsampling model) writes to `<traj_dir>/model/`; phase 2 (physics refinement) writes to `<traj_dir>/model/phase2/`. Each checkpoint contains: `weights.jld2` (parameters and state), `train_log.jld2` (loss history), `config.json` (hyperparameters and sensor locations). Optimizer state is deliberately excluded.
_Avoid_: model save, snapshot, serialized model

## Relationships

- The **KF solver** produces a **trajectory** (after **spin-up**)
- A **trajectory** is split at the **training horizon**: snapshots before it are training data, snapshots after are eval data
- The **sensor array** defines where velocity **observations** are taken from each snapshot
- The **generative model** is a two-stage pipeline: **upsampling model** → **physics refinement**
- The **upsampling model** (`VortFourierDecoder` or `ConvDecoder`) maps a **latent vector** to an approximate vorticity field
- **Relaxation** (the current **physics refinement** implementation) runs the **KF solver** from the upsampling model output for `T_relax` time units to produce the final vorticity field
- The **SWD loss** is always computed on the final output (after relaxation when enabled)
- Training runs in two phases: phase 1 trains the **upsampling model** with SWD on its direct output; phase 2 (optional) trains through the **relaxation** step with SWD on the relaxed output
- **Eval SWD** tracks generalisation against the held-out eval snapshots in physical space
- After each training phase, a **checkpoint** records weights, loss history, and configuration; phase 1 and phase 2 checkpoints live in separate directories

## Example dialogue

> **Dev:** "How do we get training data?"
> **Domain expert:** "Run the **KF solver** through **spin-up**, then record a **trajectory**. We sample **observations** from that **trajectory** and train the **generative model** against them."

> **Dev:** "What does the **generative model** output?"
> **Domain expert:** "A vorticity field — but it's two stages. The **upsampling model** produces a rough field from a **latent vector**, then **relaxation** runs the **KF solver** for a short time to push that field onto the attractor."

> **Dev:** "Why run the solver again if you already have training data from it?"
> **Domain expert:** "The **upsampling model** can produce fields that are physically implausible — off the attractor. A short **relaxation** decays those transients cheaply. The SWD loss on the relaxed output teaches the upsampling model to produce better initial conditions."

> **Dev:** "How do we know the model is learning?"
> **Domain expert:** "The **sliced Wasserstein distance** drops during training and the **eval SWD** against held-out snapshots should decrease too. After each phase, the **checkpoint** lets us regenerate plots and compare distributions before and after relaxation."
