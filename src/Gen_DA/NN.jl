module NN
export VortFourierDecoder, eval_decoder_vort, eval_decoder_vel, loss_fn, DataPipeline
include("NN/model.jl")
include("NN/loss_fn.jl")
include("NN/data_pipeline.jl")

end
