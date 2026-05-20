using Test
using Gen_DA.NN
using Lux
using Random

function make_test_inputs(rng)
    model, ps, st = StreamFourierDecoder([4, 8, 16], 4, rng)
    train_losses = Float32[0.5, 0.4, 0.3]
    eval_swds    = Float32[0.2, 0.1]
    eval_epochs  = [1, 2]
    config = Dict{String,Any}(
        "Re" => 100, "N" => 64, "layers" => [4, 8, 16],
        "num_freq" => 4, "lr" => 1e-3, "batch_size" => 8,
        "T_train" => 500.0, "n_epochs" => 10, "eval_every" => 5,
        "n_meas_space" => 20, "lr_schedule" => "constant",
        "sensor_locations" => [CartesianIndex(2, 3), CartesianIndex(5, 7)],
    )
    ps, st, train_losses, eval_swds, eval_epochs, config
end

@testset "save_checkpoint: three files are created" begin
    rng = MersenneTwister(1)
    ps, st, train_losses, eval_swds, eval_epochs, config = make_test_inputs(rng)
    mktempdir() do dir
        NN.Checkpoint.save_checkpoint(dir, ps, st, train_losses, eval_swds, eval_epochs, config)
        @test isfile(joinpath(dir, "weights.jld2"))
        @test isfile(joinpath(dir, "train_log.jld2"))
        @test isfile(joinpath(dir, "config.json"))
    end
end

@testset "save_checkpoint: config.json has expected keys and sensor locations" begin
    using JSON3
    rng = MersenneTwister(3)
    ps, st, train_losses, eval_swds, eval_epochs, config = make_test_inputs(rng)
    mktempdir() do dir
        NN.Checkpoint.save_checkpoint(dir, ps, st, train_losses, eval_swds, eval_epochs, config)
        cfg = JSON3.read(read(joinpath(dir, "config.json"), String))
        for k in ("Re","N","layers","num_freq","lr","batch_size","T_train",
                  "n_epochs","eval_every","n_meas_space","lr_schedule","sensor_locations")
            @test haskey(cfg, k)
        end
        locs = cfg["sensor_locations"]
        @test length(locs) == 2
        @test locs[1] == [2, 3]
        @test locs[2] == [5, 7]
    end
end

@testset "save_checkpoint: params round-trip through JLD2" begin
    using JLD2
    rng = MersenneTwister(2)
    ps, st, train_losses, eval_swds, eval_epochs, config = make_test_inputs(rng)
    mktempdir() do dir
        NN.Checkpoint.save_checkpoint(dir, ps, st, train_losses, eval_swds, eval_epochs, config)
        ps2 = load(joinpath(dir, "weights.jld2"), "ps")
        @test ps == ps2
    end
end
