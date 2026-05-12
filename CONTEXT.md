# gen_DA_project

A Julia codebase that generates training data and trains a generative model to learn the distribution of 2D Kolmogorov-forced Navier-Stokes flow states, with the long-term goal of data assimilation from sparse observations.

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
The neural network (`VortFourierDecoder`) that maps a latent vector to a vorticity field. Trained to reproduce the distribution of vorticity fields consistent with observations.
_Avoid_: decoder, prior, surrogate model

**Latent Vector**:
The input to the generative model, drawn from (or optimized within) a Gaussian distribution. Denoted `x` in code.
_Avoid_: latent code, latent variable, noise vector

**Spin-up**:
An initial phase of KF solver integration (duration `T_spinup`) that is discarded to allow transients to decay before recording a trajectory.
_Avoid_: burn-in, equilibration, warm-up

**Sliced Wasserstein Distance (SWD)**:
The training and evaluation loss. Projects sample sets onto random unit directions and averages the 1-D Wasserstein distance. Used in two forms: observation-space SWD (velocity at sensor locations, trained end-to-end via XLA/Enzyme) and spectral SWD (truncated Fourier coefficients of vorticity, used for eval only).
_Avoid_: Wasserstein loss, Earth mover's distance, OT loss

**Eval SWD**:
The spectral-space sliced Wasserstein distance computed every `eval_every` epochs against held-out vorticity snapshots. Measures generalisation to unseen flow states without requiring observations.
_Avoid_: validation loss, test loss

**Checkpoint**:
The saved model artefacts written to `<traj_dir>/model/` after training: `weights.jld2` (parameters and state), `train_log.jld2` (loss history), `config.json` (hyperparameters and sensor locations). Optimizer state is deliberately excluded.
_Avoid_: model save, snapshot, serialized model

## Relationships

- The **KF solver** produces a **trajectory** (after **spin-up**)
- A **trajectory** is split at the **training horizon**: snapshots before it are training data, snapshots after are eval data
- The **sensor array** defines where velocity **observations** are taken from each snapshot
- The **generative model** takes a **latent vector** and outputs a vorticity field
- The **generative model** is trained so its output distribution matches the **observations** at the **sensor array** via the **sliced Wasserstein distance**
- **Eval SWD** tracks generalisation against the held-out eval snapshots in spectral space
- After training, a **checkpoint** records weights, loss history, and configuration for reproducibility

## Example dialogue

> **Dev:** "How do we get training data?"
> **Domain expert:** "Run the **KF solver** through **spin-up**, then record a **trajectory**. We sample sparse velocity **observations** from that **trajectory** and train the **generative model** to reproduce them."

> **Dev:** "What does the **generative model** output?"
> **Domain expert:** "**Vorticity** — specifically its Fourier coefficients. We derive velocity from that when we need to compare against **observations**."

> **Dev:** "How do we know the model is learning?"
> **Domain expert:** "Two signals: the observation-space **sliced Wasserstein distance** drops during training, and the **eval SWD** against held-out snapshots should decrease too. After training, the **checkpoint** lets us regenerate plots offline."
