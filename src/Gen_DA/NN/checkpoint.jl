module Checkpoint

using JLD2
using JSON3
using Lux

export save_checkpoint

function save_checkpoint(dir, ps, st, train_losses, eval_swds, eval_epochs, config::Dict)
    mkpath(dir)

    cpu = Lux.cpu_device()
    jldsave(joinpath(dir, "weights.jld2"); ps=cpu(ps), st=cpu(st))

    jldsave(joinpath(dir, "train_log.jld2");
        train_losses=train_losses,
        eval_swds=eval_swds,
        eval_epochs=eval_epochs)

    serializable = Dict{String,Any}()
    for (k, v) in config
        if k == "sensor_locations"
            serializable[k] = [[ci[1], ci[2]] for ci in v]
        else
            serializable[k] = v
        end
    end
    open(joinpath(dir, "config.json"), "w") do io
        JSON3.write(io, serializable)
    end
end

end
