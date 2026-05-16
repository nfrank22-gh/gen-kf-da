using Random

mutable struct BatchSampler
    rng::AbstractRNG
    batch_size::Int
    n_full::Int
    n_slices::Int
    flat_dim::Int
    fix_thetas::Bool
    _thetas_cache::Union{Nothing,Matrix{Float32}}
end

function BatchSampler(rng::AbstractRNG, batch_size::Int,
                      n_full::Int, n_slices::Int, flat_dim::Int;
                      fix_thetas::Bool=true)
    BatchSampler(rng, batch_size, n_full, n_slices, flat_dim, fix_thetas, nothing)
end

function sample_epoch!(s::BatchSampler)
    if !(s.fix_thetas && s._thetas_cache !== nothing)
        arr = randn(s.rng, Float32, s.n_slices * s.n_full, s.flat_dim)
        s._thetas_cache = arr
    end
    return nothing
end

# Returns (thetas_batch, cols) for batch i. Must be called after sample_epoch!.
function get_batch(s::BatchSampler, i::Int)
    cols       = (i-1)*s.batch_size+1 : i*s.batch_size
    theta_rows = (i-1)*s.n_slices+1   : i*s.n_slices
    return s._thetas_cache[theta_rows, :], cols
end
