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
The curl of the velocity field; the primary variable of the KF solver. Stored in spectral space as Fourier coefficients (`omega_hat`) and in physical space as a real array (`omega`). Velocity `(u, v)` is derived from vorticity via the stream function. Within the generative model, vorticity is a *derived* quantity computed from the model's stream function output via `ω̂ = |k|²·ψ̂`; it is not produced directly.
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
The neural network that maps a latent vector to a divergence-free physical-space velocity field `(u, v)`. Implemented as an **upsampling model** (`StreamFourierDecoder` or `ConvDecoder`). Velocity and vorticity are derived from the model's **stream function** output analytically in spectral space.
_Avoid_: decoder, prior, surrogate model

**Upsampling Model**:
The neural network at the core of the generative model. Maps a latent vector to a **stream function** `ψ` at `N×N` resolution; divergence-free velocity `(u, v)` is derived from `ψ` analytically in spectral space (`û = iky·ψ̂`, `v̂ = -ikx·ψ̂`). Two architectures are available: `StreamFourierDecoder` and `ConvDecoder`.
_Avoid_: decoder, generative model

**StreamFourierDecoder**:
The MLP-based upsampling model architecture. An MLP maps the latent vector to free Fourier coefficients `ψ̂` of the stream function (scalar, DC mode zeroed), which are spectrally padded to the target resolution. Velocity and vorticity are derived analytically: `û = iky·ψ̂`, `v̂ = -ikx·ψ̂`, `ω̂ = |k|²·ψ̂`. Incompressibility is guaranteed by construction with no projection step.
_Avoid_: Fourier decoder, spectral decoder, VelFourierDecoder, VortFourierDecoder

**SpectralCircConv**:
A hybrid conv layer used throughout `ConvDecoder` (replacing plain `CircConv` in all dense and main convs, and both tail convs). Each call runs two parallel branches and **sums** their outputs: (1) **Conv branch** — the existing `CircConv` with circular padding (captures local spatial structure). (2) **Spectral branch** — truncated FNO-style: `rfft` the input, apply a learned complex `(k_max, k_max, C_out, C_in)` channel-mixing matrix to the two corner regions of the 2D frequency grid (`[1:k_max, 1:k_max]` and `[1:k_max, W-k_max+1:W]`), zero-pad the rest, then `irfft` back to physical space (captures long-range/global structure via the lowest `k_max` modes). Complex weights are stored as split Float32 pairs `(W_lo_re, W_lo_im, W_hi_re, W_hi_im)` so all Lux parameters remain plain `Float32`. `k_max` is configured per-block via `spectral_modes::Vector{Int}` and separately for the tail via `tail_spectral_modes::Int`.
_Avoid_: FNO layer, spectral conv, Fourier layer

**ConvDecoder**:
The conv-based upsampling model architecture. A **Fourier Base** FC maps `z` to re+im Fourier coefficients of `C` channels at a `(k_base+1)×(2·k_base)` rfft spectrum; `irfft` gives a `(2·k_base)×(2·k_base)×C` physical feature map. A cascade of `B` **UpsampleBlocks** doubles resolution at each step until `N_conv×N_conv` is reached (`2·k_base·2^B = N_conv`). A two-**SpectralCircConv** tail (`tail_conv1 → act → tail_conv2`) collapses to the single-channel stream function `ψ`. If `N_conv < N`, iterated `spectral_upsample_2x` steps bring `ψ` to `N×N`. Velocity and vorticity are derived analytically from `ψ̂`. Per-block kernel sizes set via `kernel_sizes::Vector{Int}` (length `B`); FNO truncation modes set via `spectral_modes::Vector{Int}` (length `B`); tail scalars `tail_kernel` and `tail_spectral_modes`.
_Avoid_: FiLM decoder, FourierFiLMDecoder

**UpsampleBlock**:
One stage of `ConvDecoder`. Structure: (1) spatial resolution is doubled according to `upsample_mode`: `spectral_upsample_2x` (`:spectral`, default), `ConvTranspose → norm → act` (`:conv_transpose`, learned), or `nearest_upsample_2x` (`:nearest`, parameter-free pixel replication — no norm or act applied after upsampling). (2) DenseNet dense block with **pre-activation BatchNorm**: the first dense conv (`SpectralCircConv(C_in → C_in)`) receives `x_up` directly with no pre-activation; each subsequent conv applies `BatchNorm(i·C_in) → act` to the concatenated tensor before `SpectralCircConv(i·C_in → C_in) → cat`, growing the tensor to `n_convs·C_in` channels. (3) `BatchNorm(n_convs·C_in) → act` — pre-activation normalisation. (4) Optional **GE-θ+ block** (`use_ge=true`): depth-wise `CircConv` gather → `Conv1×1` bottleneck → `act` → `Conv1×1` → sigmoid → `x * (1 + sigmoid(context))`. (5) `Conv1×1(n_convs·C_in → C_out)` — projection. No additional norms or activations between the GE block and the projection. No additive skip; gradient flow is through dense concatenation paths. Edge case: `n_convs=1` — zero dense layers; `BatchNorm(C_in) → act` precedes the optional GE then `proj`. All `SpectralCircConv` layers in a block share the same `k_max` (from `spectral_modes[i]`). GE gather kernel is set per-block via `ge_kernel_sizes[i]`; reduction ratio via `ge_reduction`. Elementwise activations (`act` and `sigmoid` in GE) are conditionally **anti-aliased** via `use_antialias` (default `true`): when enabled, every 4D spatial activation site (dense loop, post-`proj_norm`, post-`proj_out_norm`, GE bottleneck, GE sigmoid, tail) is wrapped as `spectral_upsample_2x → act → spectral_downsample_2x`; when disabled, plain `act.(x)` is used. FC/MLP activations in the latent FC path are never wrapped. See ADR 0016, ADR 0017, ADR 0018.
_Avoid_: upsampling block, conv block, FiLM block

**Stream Function**:
The scalar field `ψ` from which the velocity field is derived: `u = ∂ψ/∂y`, `v = -∂ψ/∂x`. In spectral space: `û = iky·ψ̂`, `v̂ = -ikx·ψ̂`, `ω̂ = |k|²·ψ̂`. Guarantees ∇·u = 0 by construction — no Leray projection is needed. The DC mode `ψ̂(0,0)` is zeroed (it has no effect on `u`, `v`, or `ω` but would waste a learnable degree of freedom). The native output representation of all upsampling model architectures.
_Avoid_: stream field, velocity potential, scalar potential

**Latent Vector**:
The input to the generative model. Denoted `z` in code. During training, drawn via the reparameterization trick from the **per-snapshot posterior**: `z = mu + exp(log_sigma) * eps`, where `eps ~ N(0,I)` is generated in the training loop and passed explicitly into the loss function. At eval time, sampled fresh from N(0,I); the learned posteriors are not used.
_Avoid_: latent code, latent variable, noise vector

**Latent Posterior**:
The per-snapshot variational distribution N(mu, exp(log_sigma)²) from which the **latent vector** is drawn via the reparameterization trick during training. Stored as two `latent_dim × n_train` parameter matrices `ps.latent_mu` and `ps.latent_log_sigma`, jointly optimized with the upsampling model weights via Adam; each column corresponds to one training snapshot and the batch loss indexes into them via `cols`.
_Avoid_: latent matrix, latent embedding, latent table

**Conditioned Posterior**:
A per-snapshot latent posterior N(mu, exp(log_sigma)²) obtained by freezing the generative model weights and optimizing a fresh `(mu, log_sigma)` pair against the SWD loss on velocity **observations** from a single snapshot, plus **KL Regularization**. Initialized from N(0,I) / 0 and optimized via Adam with AutoEnzyme on GPU. Drawing samples from the conditioned posterior yields plausible flow states consistent with sparse observations.
_Avoid_: posterior inference, amortized posterior

**Snapshot DA**:
A data assimilation experiment that conditions the **generative model** on velocity **observations** from a single eval-set snapshot: freeze the model weights, optimize a fresh **Conditioned Posterior** `(mu, log_sigma)` via Adam + SWD + **KL Regularization**, then draw samples and plot. Contrasts with sequential DA, which assimilates observations across many time steps. Implemented in `scripts/snapshot_DA.jl`; output written to `<checkpoint_dir>/snapshot_da/snap_{idx}/`.
_Avoid_: single-step DA, instantaneous DA, static DA

**KL Regularization**:
A penalty added to the SWD loss that encourages the **latent posterior** to stay close to the prior N(0,I). Computed as `kl_weight * 0.5 * mean(mean(sigma² + mu² - 1 - 2*log_sigma, dims=1))` where `sigma = exp.(log_sigma)`, the inner mean is over the latent dimension, and the outer mean is over the batch. Using `mean` (not `sum`) over the latent dimension makes `kl_weight` invariant to `latent_dim` — doubling `latent_dim` does not change the scale of the penalty. `kl_weight` is set in `train_ctrl.jl` and applied at full strength from epoch 1.
_Avoid_: KL divergence loss, VAE regularization, latent regularization, L2 regularization


**H2 Vorticity Regularization**:
A physics-motivated smoothness penalty added to the training loss. Penalizes large second spatial derivatives in the generated vorticity field, motivated by the viscous diffusion term in the Navier-Stokes equations (which damps modes in proportion to `|k|²`). Computed spectrally as `h2_weight * mean(|k|⁴ · |ω̂(k)|²)` where the mean is taken over all spectral modes and the batch — equivalently `h2_weight * mean(model.grid.lap.^2 .* abs2.(omega_hat))`. Operates on standardized vorticity (the model's native output space) so that `h2_weight` remains interpretable across different flow amplitudes. `omega_hat` is computed at the model's native spectral resolution (NDOF for `StreamFourierDecoder`, N for `ConvDecoder`). Applied in both VorticityMode and ObservationsMode. Defaults to `0` (disabled); set via `h2_weight` in `train_ctrl.jl`.
_Avoid_: Sobolev loss, spectral smoothness penalty, Laplacian regularization

**Latent LR Multiplier**:
A scalar (default 10) by which the learning rate for the latent variational parameters — `latent_mu`, `latent_log_sigma` — exceeds the global decoder learning rate. Applied by setting a higher `eta` on those subtrees of the Optimisers.jl state tree after optimizer setup, and re-applied after every LR scheduler step to preserve the ratio as the global LR decays. Compensates for the sparse gradient signal each per-snapshot column receives relative to the shared decoder weights.
_Avoid_: per-parameter LR, parameter group LR

**Symmetry Reduction**:
A preprocessing step applied to every vorticity snapshot immediately after loading the trajectory, before the train/eval split. Maps each snapshot to its canonical representative in the quotient space by eliminating: (1) the 8-fold discrete shift-reflect symmetry S (y-shift by π/n + sign flip) using the method-of-symmetry-charting — sector = floor(arg(ω̂(0,1)) / (π/n)), correction S^k with k = (n−1)·sector mod 2n applied in physical space; (2) the 2-fold rotation symmetry R (spatial rotation by π = complex conjugation of ω̂) by checking sign of imag(ω̂(0,n)); (3) the continuous streamwise (x) translation symmetry via the first Fourier mode method of slices — aligning ω̂(1,0) to the positive real axis. Implemented as `reduce_symmetries` / `reduce_trajectory` in `DataPipeline`. Replaces the learned **Periodic Translation Parameters** approach; the model now outputs in the reduced space and no shift correction is needed at eval time.
_Avoid_: symmetry augmentation, symmetry equivariance, data normalisation

**Spin-up**:
An initial phase of KF solver integration (duration `T_spinup`) that is discarded to allow transients to decay before recording a trajectory.
_Avoid_: burn-in, equilibration, warm-up

**Sliced Wasserstein Distance (SWD)**:
A training and evaluation loss. Projects sample sets onto random unit directions and averages the 1-D Wasserstein distance. Always computed in physical space: either over flattened vorticity fields (`N×N`) or over velocity observations at sensor locations. Selected via `recon_loss = :swd`.
_Avoid_: Wasserstein loss, Earth mover's distance, OT loss

**Sinkhorn Divergence**:
An alternative reconstruction loss to SWD. Computes the debiased entropic optimal transport divergence `S_ε(P,Q) = OT_ε(P,Q) − ½·OT_ε(P,P) − ½·OT_ε(Q,Q)`. The cost matrix is normalised by `mean(C_PQ)` before Sinkhorn iterations so that `sinkhorn_eps` is scale-invariant. Fixed iteration count (`sinkhorn_n_iter`, default 100) for XLA compatibility. Selected via `recon_loss = :sinkhorn`; configured via `sinkhorn_eps`, `sinkhorn_n_iter`, and `sinkhorn_metric` in `train_ctrl.jl`. Four ground metrics are available: `:sq_l2` (squared Euclidean), `:cosine` (cosine dissimilarity `1 − cosθ`), `:lr_mahalanobis` (Low-Rank Mahalanobis — squared L2 in the **Mahalanobis Projection** subspace), and `:rand_lr_mahalanobis` (squared L2 in a fresh **Random Projection** subspace regenerated each epoch). The **Eval SWD** metric remains SWD regardless of which reconstruction loss is used for training.
_Avoid_: regularised OT, entropic OT, Sinkhorn loss

**Mahalanobis Projection**:
A `(feature_dim × r)` matrix `L` whose columns are the top-`r` PCA directions of the standardized training set. Used as the ground metric for **Sinkhorn Divergence** when `sinkhorn_metric = :lr_mahalanobis`: the pairwise cost is `d²(x,y) = ‖L⊤x − L⊤y‖²`. Computed from standardized training snapshots after the train/eval split — VorticityMode uses flattened vorticity (`N²`-dim), ObservationsMode uses concatenated `(u,v)` observations at the fixed sensor array (`2·n_meas`-dim, not available for random sensors). Rank `r` is set via `mahalanobis_rank` in `train_ctrl.jl`. Not saved in checkpoints; recomputed from training data on each run.
_Avoid_: PCA metric, low-rank metric, Mahalanobis matrix

**Random Projection**:
A `(feature_dim × r)` matrix `L` of iid Gaussian entries drawn fresh at the start of each epoch. Used as the ground metric for **Sinkhorn Divergence** when `sinkhorn_metric = :rand_lr_mahalanobis`: the pairwise cost is `d²(x,y) = ‖L⊤x − L⊤y‖²`, same formula as **Mahalanobis Projection** but with a random basis instead of PCA. Generated inside `prepare_epoch_fn` (once per epoch, same `L` for all batches in that epoch); no pre-training PCA step required. The per-epoch randomization mitigates the curse of dimensionality in high-dimensional feature spaces and injects gradient noise that acts as a regularizer. Rank `r` is shared with `mahalanobis_rank` in `train_ctrl.jl`.
_Avoid_: Johnson-Lindenstrauss projection, random Mahalanobis

**Snapshot Standardization**:
Per-pixel z-scoring applied to all snapshots after symmetry reduction and the train/eval split. The mean field `μ[i,j]` and std field `σ[i,j]` are computed from training snapshots only; the same statistics are applied to eval snapshots to avoid data leakage. The model trains and outputs vorticity in standardized space (mean 0, unit per-pixel variance). Un-standardization (`ω_phys = ω_std .* σ .+ μ`) is applied before diagnostic plots. Stats are saved in `config.json` for checkpoint reproducibility. Implemented as `standardize_snapshots` (computes stats, returns standardized train data) and `apply_standardization` (applies pre-computed stats) in `DataPipeline`.
_Avoid_: normalization, whitening, data normalization

**Eval SWD**:
The physical-space sliced Wasserstein distance computed every `eval_every` epochs against held-out vorticity snapshots (flattened `N×N` arrays). Measures generalisation to unseen flow states without requiring observations. Always computed with SWD regardless of the training reconstruction loss.
_Avoid_: validation loss, test loss

**Checkpoint**:
The saved model artefacts written to `<traj_dir>/model/` after training. Contains: `weights.jld2` (parameters and state), `train_log.jld2` (loss history), `config.json` (hyperparameters and sensor locations). Optimizer state is deliberately excluded.
_Avoid_: model save, snapshot, serialized model

## Relationships

- The **KF solver** produces a **trajectory** (after **spin-up**)
- A **trajectory** is split at the **training horizon**: snapshots before it are training data, snapshots after are eval data
- The **sensor array** defines where velocity **observations** are taken from each snapshot
- The **generative model** is the **upsampling model** (`StreamFourierDecoder` or `ConvDecoder`): it maps a **latent vector** to a **stream function** `ψ`; divergence-free `(u, v)` and vorticity are derived analytically from `ψ̂` in spectral space
- The **SWD loss** is computed on `(u, v)` output derived from `ψ`; vorticity is derived from `ψ̂` when needed for **VorticityMode** or diagnostic plots
- **Eval SWD** tracks generalisation against the held-out eval snapshots in physical space
- After training, a **checkpoint** records weights, loss history, and configuration

## Example dialogue

> **Dev:** "How do we get training data?"
> **Domain expert:** "Run the **KF solver** through **spin-up**, then record a **trajectory**. We sample **observations** from that **trajectory** and train the **generative model** against them."

> **Dev:** "What does the **generative model** output?"
> **Domain expert:** "A divergence-free velocity field. The **upsampling model** produces a **stream function** `ψ` from a **latent vector**; velocity `(u, v)` and vorticity are derived from `ψ` analytically in spectral space."

> **Dev:** "How do we know the model is learning?"
> **Domain expert:** "The **sliced Wasserstein distance** drops during training and the **eval SWD** against held-out snapshots should decrease too. In phase 2 the eval runs the full pipeline — upsampling model then neural operator — so the metric reflects what the combined generative model produces."
