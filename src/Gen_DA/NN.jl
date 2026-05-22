module NN
import ..SpectralGrid, ..spectral_pad, ..velocity_from_psi_hat, ..vorticity_from_vel
export SpectralGrid, spectral_pad, velocity_from_psi_hat, vorticity_from_vel,
       StreamFourierDecoder, ConvDecoder, eval_decoder_vort, eval_decoder_vel,
       loss_fn, loss_fn_vort_state,
       sliced_wasserstein, kl_regularization,
       DataPipeline, Checkpoint, TrainingPlots, plot_conditioned_vorticity,
       build_optimizer, ReduceOnPlateau, step!,
       BatchSampler, sample_epoch!, get_batch,
       VorticityMode, ObservationsMode,
       TrainingConfig, TrainingSession, train!,
       build_model_from_config
include("NN/model.jl")
include("NN/conv_decoder.jl")
include("NN/loss_fn.jl")
include("NN/data_pipeline.jl")
include("NN/checkpoint.jl")
include("NN/training_utils.jl")
include("NN/training_plots.jl")
include("NN/batch_sampler.jl")
include("NN/training_mode.jl")
include("NN/training_session.jl")
include("NN/model_factory.jl")

end
