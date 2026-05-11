using Optimisers

export build_optimizer

function build_optimizer(lr::AbstractFloat, n_epochs::Int, use_cosine_lr::Bool)
    opt = Adam(lr)
    use_cosine_lr || return opt, nothing
    T = typeof(lr)
    schedule = n -> T(lr) * (1 + cos(T(π) * (n - 1) / n_epochs)) / 2
    return opt, schedule
end
