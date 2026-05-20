using Lux

# Reconstruct the model from a saved config dict.
# Returns only the model; load ps and st separately from the checkpoint.
# Activation is fixed to gelu — it is not serialised in config.json.
# If the config contains encoder_hidden, returns an ObservationEncoderDecoder.
function build_model_from_config(config::Dict, rng)
    T          = Float32
    N          = Int(config["N"])
    latent_dim = Int(config["latent_dim"])
    arch       = Symbol(config["model_arch"])

    if arch == :fourier
        num_freq = Int(config["num_freq"])
        layers   = Vector{Int}(config["layers"])
        decoder, _, _ = StreamFourierDecoder(layers, num_freq, rng, T)

    elseif arch == :conv
        decoder, _, _ = ConvDecoder(
            latent_dim,
            Vector{Int}(config["fc_hidden"]),
            Int(config["k_base"]),
            Int(config["init_channels"]),
            Vector{Int}(config["conv_channels"]),
            Int(config["n_convs_per_block"]),
            Vector{Int}(config["kernel_sizes"]),
            Int(config["tail_kernel"]),
            gelu,
            Int(config["N_conv"]),
            N, rng, T)

    else
        error("Unknown model_arch: $(config["model_arch"])")
    end

    if haskey(config, "encoder_hidden")
        encoder_hidden      = Vector{Int}(config["encoder_hidden"])
        encoder_head_hidden = Vector{Int}(config["encoder_head_hidden"])
        encoder, _, _ = DeepSetsEncoder(encoder_hidden, encoder_head_hidden, latent_dim, rng, T)
        return ObservationEncoderDecoder(encoder, decoder)
    end

    return decoder
end
