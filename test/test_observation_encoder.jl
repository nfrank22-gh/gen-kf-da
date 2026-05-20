using Test, Gen_DA, Gen_DA.NN, Random, Lux

@testset "DeepSetsEncoder shapes" begin
    rng        = Xoshiro(42)
    latent_dim = 8
    n_meas     = 12
    B          = 5

    encoder, ps, st = DeepSetsEncoder([64, 128], [64], latent_dim, rng, Float32)

    obs = randn(rng, Float32, 6, n_meas, B)
    mu, log_sigma, new_st = encoder(obs, ps, st)

    @test size(mu)        == (latent_dim, B)
    @test size(log_sigma) == (latent_dim, B)
end

@testset "ObservationEncoderDecoder encode/decode shapes" begin
    rng        = Xoshiro(7)
    T          = Float32
    N          = 16
    latent_dim = 8
    n_meas     = 10
    B          = 4

    decoder, dec_ps, dec_st = ConvDecoder(
        latent_dim, [32], 2, 4, [8, 4], 1, [3, 3], 3, gelu, N, N, rng, T)

    encoder, enc_ps, enc_st = DeepSetsEncoder([32, 64], [32], latent_dim, rng, T)

    model = ObservationEncoderDecoder(encoder, decoder)
    ps    = (encoder=enc_ps, decoder=dec_ps)
    st    = (encoder=enc_st, decoder=dec_st)

    # Encode: observations → (mu, log_sigma)
    obs = randn(rng, T, 6, n_meas, B)
    mu, log_sigma, _ = encode(model, obs, ps, st)
    @test size(mu)        == (latent_dim, B)
    @test size(log_sigma) == (latent_dim, B)

    # Decode: z → vorticity (bypasses encoder)
    z = randn(rng, T, latent_dim, B)
    omega, _ = eval_decoder_vort(model, N, z, ps, st)
    @test size(omega) == (N, N, B)

    # Decode: z → velocity
    u, v, _ = eval_decoder_vel(model, N, z, ps, st)
    @test size(u) == (N, N, B)
    @test size(v) == (N, N, B)
end

@testset "_sensor_fourier_features shape and periodicity" begin
    N          = 32
    n_meas     = 20
    sensor_lin = rand(1:N*N, n_meas)

    sf = Gen_DA.NN._sensor_fourier_features(sensor_lin, N)
    @test size(sf) == (4, n_meas)
    # All values in [-1, 1] (sin/cos outputs)
    @test all(-1 .<= sf .<= 1)
end

@testset "_sensor_fourier_features_per_sample shape" begin
    N          = 32
    n_meas     = 15
    B          = 6
    sensor_lin = Int32.(rand(1:N*N, n_meas, B))

    sf = Gen_DA.NN._sensor_fourier_features_per_sample(sensor_lin, N)
    @test size(sf) == (4, n_meas, B)
    @test all(-1 .<= sf .<= 1)
end
