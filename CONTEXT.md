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
The convolutional upsampling model architecture. FC layers map the latent vector to a small spatial feature map, which is progressively upsampled to `N_conv×N_conv` via a series of spectral upsampling blocks, then collapsed to a 1-channel vorticity field by a final convolution. If `N_conv < N`, a final Fourier interpolation step (repeated spectral 2× upsampling) brings the output to the full `N×N` training resolution. `N / N_conv` must be a power of 2. When `N_conv = N` no interpolation step is applied and the behaviour is identical to the original architecture.
_Avoid_: conv decoder, upsampling decoder

**Physics Refinement**:
The second stage of the generative model. A learned dynamical process that iteratively refines the upsampling model's output, projecting it toward the system's attractor. Implemented as a **Neural Operator** applied autoregressively for `n_fno_steps` iterations. The KF solver is not used in phase 2.
_Avoid_: fine-tuning, post-processing, correction step, relaxation, solver rollout

**Neural Operator**:
The FNO-based implementation of physics refinement. A Fourier Neural Operator (FNO) that maps a vorticity field (`N×N`) to a refined vorticity field (`N×N`) in a single forward pass. Applied autoregressively `n_fno_steps` times to the upsampling model output. Trained from scratch during phase 2 jointly with the upsampling model via a single end-to-end SWD loss. Hyperparameters: `n_modes` (spectral truncation, default = `num_freq`), `n_fno_layers` (depth, default = 4), `fno_channels` (hidden width, default = 32).
_Avoid_: relaxation, solver surrogate, dynamics model

**Spectral Upsampling Block**:
One stage of the `ConvDecoder`. Doubles spatial resolution via spectral interpolation (FFT zero-pad → IFFT), applies `n_convs_per_block` circular-padded convolutions at constant channel width, adds the upsampled input as a skip connection, then reduces channels via a 1×1 convolution.
_Avoid_: upsampling layer, decoder block

**Latent Vector**:
The input to the generative model. Denoted `z` in code. During training, drawn via the reparameterization trick from the **per-snapshot posterior**: `z = mu + exp(log_sigma) * eps`, where `eps ~ N(0,I)` is generated in the training loop and passed explicitly into the loss function. At eval time, sampled fresh from N(0,I); the learned posteriors are not used.
_Avoid_: latent code, latent variable, noise vector

**Latent Posterior**:
The per-snapshot variational distribution N(mu, exp(log_sigma)²) stored as two `latent_dim × n_train` parameter matrices: `ps.latent_mu` (initialized from N(0,I)) and `ps.latent_log_sigma` (initialized to 0, so sigma starts at 1). Both are jointly optimized with the upsampling model weights via Adam. Each column corresponds to one training snapshot; the batch loss indexes into them via `cols`. When `fix_latents_phase2` is true, both matrices are frozen together via `Optimisers.freeze!`.
_Avoid_: latent matrix, latent embedding, latent table, encoder

**KL Regularization**:
A penalty added to the SWD loss that encourages the **latent posterior** to stay close to the prior N(0,I). Computed as `kl_weight * 0.5 * mean(sum(sigma² + mu² - 1 - 2*log_sigma, dims=1))` where `sigma = exp.(log_sigma)` and the sum is over the latent dimension. This is the standard KL from N(mu, sigma²) to N(0,I), averaged over the batch. Weight `kl_weight` is the beta scalar set in `train_ctrl.jl`.
_Avoid_: KL divergence loss, VAE regularization, latent regularization, L2 regularization

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
- **Physics refinement** (the **neural operator**) applies the FNO autoregressively `n_fno_steps` times to the upsampling model output to produce the final vorticity field
- The **SWD loss** is always computed on the final output (after physics refinement when enabled)
- Training runs in two phases: phase 1 trains the **upsampling model** with SWD on its direct output; phase 2 (optional) trains the combined upsampling model + neural operator end-to-end with SWD on the refined output
- **Eval SWD** tracks generalisation against the held-out eval snapshots in physical space
- After each training phase, a **checkpoint** records weights, loss history, and configuration; phase 1 and phase 2 checkpoints live in separate directories

## Example dialogue

> **Dev:** "How do we get training data?"
> **Domain expert:** "Run the **KF solver** through **spin-up**, then record a **trajectory**. We sample **observations** from that **trajectory** and train the **generative model** against them."

> **Dev:** "What does the **generative model** output?"
> **Domain expert:** "A vorticity field — but it's two stages. The **upsampling model** produces a rough field from a **latent vector**, then the **neural operator** refines it autoregressively `n_fno_steps` times to push it toward the attractor."

> **Dev:** "Why add a neural operator if you already have training data from the KF solver?"
> **Domain expert:** "The **upsampling model** can produce fields that are physically implausible — off the attractor. The **neural operator** learns to refine those fields during phase-2 training. Unlike the KF solver, it's fully differentiable, so gradients flow end-to-end through both stages with a single SWD loss."

> **Dev:** "How do we know the model is learning?"
> **Domain expert:** "The **sliced Wasserstein distance** drops during training and the **eval SWD** against held-out snapshots should decrease too. In phase 2 the eval runs the full pipeline — upsampling model then neural operator — so the metric reflects what the combined generative model produces."
