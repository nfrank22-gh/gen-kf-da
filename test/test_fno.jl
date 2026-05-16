using Test
import Gen_DA
using Gen_DA.NN
using FFTW
using Random

@testset "FNO" begin
    N          = 16
    n_modes    = 4
    channels   = 8
    n_layers   = 2
    batch_size = 3
    rng        = Xoshiro(42)

    @testset "FNOLayer shape and type" begin
        layer = FNOLayer(N, n_modes, channels, channels)
        ps, st = Lux.setup(rng, layer)
        # ps from initialparameters override: bypass, R_real, R_imag
        @test haskey(ps, :R_real)
        @test haskey(ps, :R_imag)
        @test size(ps.R_real) == (channels, channels)

        v   = randn(rng, Float32, N, N, channels, batch_size)
        out, new_st = layer(v, ps, st)
        @test size(out) == (N, N, channels, batch_size)
        @test eltype(out) == Float32
    end

    @testset "FourierNeuralOperator shape and residual" begin
        fno        = FourierNeuralOperator(N, n_modes, channels, n_layers)
        ps, st     = Lux.setup(rng, fno)
        omega      = randn(rng, Float32, N, N, batch_size)
        out, new_st = fno(omega, ps, st)
        @test size(out) == (N, N, batch_size)
        @test eltype(out) == Float32
        # residual: output should differ from input
        @test !isapprox(out, omega)
    end

    @testset "UpsamplerWithFNO with VortFourierDecoder" begin
        latent_dim = 10
        num_freq   = 4
        layers     = [latent_dim, 32, 64]
        upsampler, ps_up, st_up = VortFourierDecoder(layers, num_freq, rng, Float32)
        n_fno_steps = 2

        combined, ps, st = UpsamplerWithFNO(
            upsampler, ps_up, st_up, N, n_modes, channels, n_layers, n_fno_steps, rng)

        @test haskey(ps, :upsampler)
        @test haskey(ps, :fno)

        x = randn(rng, Float32, latent_dim, batch_size)
        omega, new_st = eval_decoder_vort(combined, N, x, ps, st)
        @test size(omega) == (N, N, batch_size)
        @test eltype(omega) == Float32
    end

    @testset "UpsamplerWithFNO with ConvDecoder" begin
        latent_dim = 10
        upsampler, ps_up, st_up = ConvDecoder(
            latent_dim, [32], 4, 2, [4, 2], 2, 3, gelu, :none, 4, N, rng, Float32)
        n_fno_steps = 2

        combined, ps, st = UpsamplerWithFNO(
            upsampler, ps_up, st_up, N, n_modes, channels, n_layers, n_fno_steps, rng)

        x = randn(rng, Float32, latent_dim, batch_size)
        omega, new_st = eval_decoder_vort(combined, N, x, ps, st)
        @test size(omega) == (N, N, batch_size)
        @test eltype(omega) == Float32
    end

    @testset "eval_decoder_vel shape" begin
        latent_dim = 10
        upsampler, ps_up, st_up = ConvDecoder(
            latent_dim, [32], 4, 2, [4, 2], 2, 3, gelu, :none, 4, N, rng, Float32)
        combined, ps, st = UpsamplerWithFNO(
            upsampler, ps_up, st_up, N, n_modes, channels, n_layers, 1, rng)

        x  = randn(rng, Float32, latent_dim, batch_size)
        u, v, new_st = eval_decoder_vel(combined, N, x, ps, st)
        @test size(u) == (N, N, batch_size)
        @test size(v) == (N, N, batch_size)
    end

    @testset "n_fno_steps > 1 changes output vs n_fno_steps = 1" begin
        latent_dim = 10
        num_freq   = 4
        layers     = [latent_dim, 32, 64]
        upsampler, ps_up, st_up = VortFourierDecoder(layers, num_freq, rng, Float32)

        m1, ps1, st1 = UpsamplerWithFNO(
            upsampler, ps_up, st_up, N, n_modes, channels, n_layers, 1, rng)
        m3, ps3, st3 = UpsamplerWithFNO(
            upsampler, ps_up, st_up, N, n_modes, channels, n_layers, 3, rng)
        # Use same FNO params for both to isolate step-count effect
        ps3_same = (upsampler=ps1.upsampler, fno=ps1.fno)
        st3_same = (upsampler=st1.upsampler, fno=st1.fno)

        x = randn(rng, Float32, latent_dim, batch_size)
        out1, _ = eval_decoder_vort(m1, N, x, ps1, st1)
        out3, _ = eval_decoder_vort(m3, N, x, ps3_same, st3_same)
        @test !isapprox(out1, out3)
    end
end
