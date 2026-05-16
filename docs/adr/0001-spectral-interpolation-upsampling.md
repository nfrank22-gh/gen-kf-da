# Spectral interpolation for spatial upsampling in ConvDecoder

The `ConvDecoder` needs to upsample feature maps from 8×8 to N×N (e.g. 128×128) across four stages. We chose spectral interpolation — FFT zero-padding in frequency space followed by IFFT — over the two natural alternatives:

- **Transposed convolution**: learns upsampling weights, but has no mechanism to respect periodic boundary conditions and introduces checkerboard artefacts that must be trained away.
- **Bilinear interpolation**: smooth and deterministic, but is an approximation that breaks periodicity at the domain boundary.

Spectral interpolation is exact for band-limited signals and is the natural choice for a doubly-periodic domain (Kolmogorov-forced NS). It is the same operation used by `VortFourierDecoder`'s `spectral_pad` when decoding at higher resolution than the model's internal grid, so the physics-consistency argument applies equally here.

**Consequence**: upsampling involves FFT operations inside the forward pass, which are only differentiable through Reactant/XLA. Training must run on GPU. This is already a hard requirement of the project.
