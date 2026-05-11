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
A sparse sample of the flow used to train or evaluate the generative model. Currently: velocity `(u, v)` at random spatial grid locations sampled from a trajectory. Future: passive tracer particle positions.
_Avoid_: measurement, data point

**Generative Model**:
The neural network (`VortFourierDecoder`) that maps a latent vector to a vorticity field. Trained to reproduce the distribution of vorticity fields consistent with observations.
_Avoid_: decoder, prior, surrogate model

**Latent Vector**:
The input to the generative model, drawn from (or optimized within) a Gaussian distribution. Denoted `x` in code.
_Avoid_: latent code, latent variable, noise vector

**Spin-up**:
An initial phase of KF solver integration (duration `T_spinup`) that is discarded to allow transients to decay before recording a trajectory.
_Avoid_: burn-in, equilibration, warm-up

## Relationships

- The **KF solver** produces a **trajectory** (after **spin-up**)
- A **trajectory** is sampled to produce **observations**
- The **generative model** takes a **latent vector** and outputs a vorticity field
- The **generative model** is trained so its output distribution matches the **observations**

## Example dialogue

> **Dev:** "How do we get training data?"
> **Domain expert:** "Run the **KF solver** through **spin-up**, then record a **trajectory**. We sample sparse velocity **observations** from that **trajectory** and train the **generative model** to reproduce them."

> **Dev:** "What does the **generative model** output?"
> **Domain expert:** "**Vorticity** — specifically its Fourier coefficients. We derive velocity from that when we need to compare against **observations**."
