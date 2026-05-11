using Test
using Gen_DA.NN
using Random

@testset "plot_train_loss_curve: creates loss_curve.png" begin
    mktempdir() do dir
        NN.TrainingPlots.plot_train_loss_curve(Float32[0.5, 0.4, 0.3, 0.2], dir)
        @test isfile(joinpath(dir, "loss_curve.png"))
        @test filesize(joinpath(dir, "loss_curve.png")) > 0
    end
end

@testset "plot_eval_swd_curve: creates eval_swd_curve.png" begin
    mktempdir() do dir
        NN.TrainingPlots.plot_eval_swd_curve(Float32[0.3, 0.2], [10, 20], dir)
        @test isfile(joinpath(dir, "eval_swd_curve.png"))
        @test filesize(joinpath(dir, "eval_swd_curve.png")) > 0
    end
end

@testset "plot_vorticity_panel: creates vorticity_panel.png" begin
    rng = MersenneTwister(1)
    N = 8
    gen_omega = randn(rng, Float32, N, N, 4)
    gt_omega  = randn(rng, Float32, N, N, 4)
    mktempdir() do dir
        NN.TrainingPlots.plot_vorticity_panel(gen_omega, gt_omega, dir)
        @test isfile(joinpath(dir, "vorticity_panel.png"))
        @test filesize(joinpath(dir, "vorticity_panel.png")) > 0
    end
end

@testset "plot_energy_spectrum: creates energy_spectrum.png" begin
    rng = MersenneTwister(2)
    nfreq = 5; NDOF = 9
    gen_hat = randn(rng, ComplexF32, nfreq, NDOF, 4)
    gt_hat  = randn(rng, ComplexF32, nfreq, NDOF, 4)
    mktempdir() do dir
        NN.TrainingPlots.plot_energy_spectrum(gen_hat, gt_hat, dir)
        @test isfile(joinpath(dir, "energy_spectrum.png"))
        @test filesize(joinpath(dir, "energy_spectrum.png")) > 0
    end
end
