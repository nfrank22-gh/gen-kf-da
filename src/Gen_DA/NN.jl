module NN
import ..SpectralGrid
export SpectralGrid,
       VortFourierDecoder, eval_decoder_vort, eval_decoder_vel,
       loss_fn, loss_fn_vort, loss_fn_vort_state, sliced_wasserstein, sliced_wasserstein_spectral,
       DataPipeline, Checkpoint, TrainingPlots,
       build_optimizer, ReduceOnPlateau, step!,
       BatchSampler, sample_epoch!, get_batch,
       TrainingSession, train!
include("NN/model.jl")
include("NN/loss_fn.jl")
include("NN/data_pipeline.jl")
include("NN/checkpoint.jl")
include("NN/training_utils.jl")
include("NN/training_plots.jl")
include("NN/batch_sampler.jl")
include("NN/training_session.jl")

end
