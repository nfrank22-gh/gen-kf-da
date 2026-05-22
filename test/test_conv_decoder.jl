using Test, Gen_DA.NN, Random, Lux

# N=32, k_base=4, 2 blocks: 2·4·2²=32=N_conv ✓
@testset "ConvDecoder shapes (2 blocks, n_convs=2)" begin
    rng   = Xoshiro(42)
    T     = Float32
    N     = 32
    batch = 3

    model, ps, st = ConvDecoder(
        16,        # latent_dim
        [32],      # fc_hidden
        4,         # k_base  (2·k_base=8; 8·2²=32=N_conv ✓)
        8,         # init_channels C
        [16, 8],   # conv_channels (2 blocks: 8→16→8)
        2,         # n_convs_per_block
        [3, 3],    # kernel_sizes (one per block)
        3,         # tail_kernel
        gelu,
        N,         # N_conv == N: no spectral upsample at end
        N, rng, T;
        spectral_modes=[4, 4], tail_spectral_modes=4)

    @test model.N_conv == N
    @test model.k_base == 4
    @test model.n_blocks == 2

    x = randn(rng, T, 16, batch)

    u, v, _ = model(x, ps, st)
    @test size(u) == (N, N, batch)
    @test size(v) == (N, N, batch)

    u2, v2, _ = eval_decoder_vel(model, N, x, ps, st)
    @test size(u2) == (N, N, batch)
    @test size(v2) == (N, N, batch)

    omega, _ = eval_decoder_vort(model, N, x, ps, st)
    @test size(omega) == (N, N, batch)
end

# N_conv < N: conv backbone stops at 16×16, spectral upsample to 32×32
# k_base=4, 1 block: 2·4·2¹=16=N_conv ✓; N ÷ N_conv = 2 = 2¹ ✓
@testset "ConvDecoder N_conv < N (spectral upsample at end)" begin
    rng   = Xoshiro(7)
    T     = Float32
    N     = 32
    batch = 2

    model, ps, st = ConvDecoder(
        8,        # latent_dim
        [16],     # fc_hidden
        4,        # k_base  (2·k_base=8; 8·2¹=16=N_conv ✓)
        4,        # init_channels C
        [8],      # conv_channels (1 block: 4→8)
        2,        # n_convs_per_block
        [3],      # kernel_sizes (one per block)
        3,        # tail_kernel
        gelu,
        16,       # N_conv=16; N ÷ N_conv = 2 ✓
        N, rng, T;
        spectral_modes=[4], tail_spectral_modes=4)

    @test model.N_conv == 16

    x = randn(rng, T, 8, batch)
    u, v, _ = model(x, ps, st)
    @test size(u) == (N, N, batch)
    @test size(v) == (N, N, batch)

    omega, _ = eval_decoder_vort(model, N, x, ps, st)
    @test size(omega) == (N, N, batch)
end

# N=16, k_base=4, single block, no fc hidden layers
# 2·4·2¹=16=N_conv=N ✓
@testset "ConvDecoder single block, no hidden layers" begin
    rng   = Xoshiro(99)
    T     = Float32
    N     = 16
    batch = 4

    model, ps, st = ConvDecoder(
        4,      # latent_dim
        Int[],  # fc_hidden — direct Dense, no hidden layers
        4,      # k_base  (2·k_base=8; 8·2¹=16=N_conv ✓)
        4,      # init_channels C
        [8],    # conv_channels (1 block: 4→8)
        1,      # n_convs_per_block (no dense layers, just main_conv)
        [3],    # kernel_sizes
        3,      # tail_kernel
        gelu,
        N,      # N_conv == N
        N, rng, T;
        spectral_modes=[4], tail_spectral_modes=4)

    x = randn(rng, T, 4, batch)
    u, v, _ = model(x, ps, st)
    @test size(u) == (N, N, batch)
    @test size(v) == (N, N, batch)
end

# Per-block kernel sizes: mixed kernels
@testset "ConvDecoder per-block kernel sizes" begin
    rng   = Xoshiro(11)
    T     = Float32
    N     = 32
    batch = 2

    model, ps, st = ConvDecoder(
        8, Int[], 4, 4, [8, 4], 2, [5, 3], 5, gelu, N, N, rng, T;
        spectral_modes=[4, 4], tail_spectral_modes=4)

    x = randn(rng, T, 8, batch)
    u, v, _ = model(x, ps, st)
    @test size(u) == (N, N, batch)
    @test size(v) == (N, N, batch)
end

# SpectralCircConv parameter structure
@testset "SpectralCircConv has W_lo/W_hi re+im params" begin
    rng = Xoshiro(5)
    T   = Float32
    N   = 16
    model, ps, st = ConvDecoder(
        4, Int[], 4, 4, [4], 1, [3], 3, gelu, N, N, rng, T;
        spectral_modes=[4], tail_spectral_modes=4)
    # tail_conv1 is SpectralCircConv; its ps should have W_lo_re etc.
    tc1_ps = ps.tail_conv1
    @test haskey(tc1_ps, :W_lo_re)
    @test haskey(tc1_ps, :W_lo_im)
    @test haskey(tc1_ps, :W_hi_re)
    @test haskey(tc1_ps, :W_hi_im)
    @test size(tc1_ps.W_lo_re) == (4, 4, 1, 4)   # (k_max, k_max, C_out, C_in)
end
