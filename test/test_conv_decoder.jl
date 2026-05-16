using Test, Gen_DA.NN, Random, Lux

@testset "ConvDecoder shapes" begin
    rng = Xoshiro(42)
    T   = Float32
    N   = 32    # small grid for CPU tests
    latent_dim    = 16
    fc_hidden     = [32]
    init_channels = 8
    n_upsample_blocks = 2   # 8×8 → 16×16 → 32×32
    conv_channels = [8, 4]
    n_convs_per_block = 2
    kernel_size   = 3
    norm_type     = :none
    n_groups      = 4
    batch         = 3

    model, ps, st = ConvDecoder(latent_dim, fc_hidden, init_channels,
                                n_upsample_blocks, conv_channels,
                                n_convs_per_block, kernel_size,
                                gelu, norm_type, n_groups, N, rng, T)

    x = randn(rng, T, latent_dim, batch)
    omega, _ = model(x, ps, st)
    @test size(omega) == (N, N, batch)

    omega2, _ = eval_decoder_vort(model, N, x, ps, st)
    @test size(omega2) == (N, N, batch)
end

@testset "ConvDecoder norm_type :group" begin
    rng = Xoshiro(7)
    T   = Float32
    N   = 32
    model, ps, st = ConvDecoder(8, Int[], 4, 2, [4, 2], 1, 3,
                                relu, :group, 2, N, rng, T)
    x = randn(rng, T, 8, 2)
    omega, _ = model(x, ps, st)
    @test size(omega) == (N, N, 2)
end
