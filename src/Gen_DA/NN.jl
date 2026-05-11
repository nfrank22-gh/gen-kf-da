module NN
export VortFourierDecoder, eval_decoder_vort, eval_decoder_vel, eval_decoder_vort_hat,
       loss_fn, sliced_wasserstein_spectral, DataPipeline, Checkpoint, build_optimizer
include("NN/model.jl")
include("NN/loss_fn.jl")
include("NN/data_pipeline.jl")
include("NN/checkpoint.jl")
include("NN/training_utils.jl")

end
