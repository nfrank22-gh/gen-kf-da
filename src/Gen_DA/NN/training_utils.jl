using Optimisers

export build_optimizer, LRScheduler, ReduceOnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts,
       step!, _adjust_latent_lr!

function build_optimizer(lr::AbstractFloat)
    Adam(lr)
end

abstract type LRScheduler end

mutable struct ReduceOnPlateau{T<:AbstractFloat} <: LRScheduler
    factor::T
    patience::Int
    min_lr::T
    current_lr::T
    best::T
    wait::Int
end

function ReduceOnPlateau(lr::T; factor::T=T(0.5), patience::Int=10, min_lr::T=T(1e-6)) where {T<:AbstractFloat}
    ReduceOnPlateau{T}(factor, patience, min_lr, lr, typemax(T), 0)
end

function step!(sched::ReduceOnPlateau, epoch::Int, metric)
    if metric < sched.best
        sched.best = metric
        sched.wait = 0
    else
        sched.wait += 1
        if sched.wait >= sched.patience
            sched.current_lr = max(sched.current_lr * sched.factor, sched.min_lr)
            sched.wait = 0
            println("  LR reduced to $(sched.current_lr)")
        end
    end
    return sched.current_lr
end

mutable struct CosineAnnealingLR{T<:AbstractFloat} <: LRScheduler
    lr_max::T
    eta_min::T
    T_max::Int
    current_lr::T
end

function CosineAnnealingLR(lr::T; T_max::Int, eta_min::T=T(1e-6)) where {T<:AbstractFloat}
    CosineAnnealingLR{T}(lr, eta_min, T_max, lr)
end

function step!(sched::CosineAnnealingLR{T}, epoch::Int, metric) where {T}
    t = min(epoch, sched.T_max)
    sched.current_lr = sched.eta_min + T(0.5) * (sched.lr_max - sched.eta_min) * (1 + cos(T(π) * t / sched.T_max))
    return sched.current_lr
end

mutable struct CosineAnnealingWarmRestarts{T<:AbstractFloat} <: LRScheduler
    lr_max::T
    eta_min::T
    T_0::Int       # initial cycle length in epochs
    T_mult::Int    # multiplier applied to cycle length at each restart
    T_cur::Int     # current cycle length
    t_cur::Int     # epochs elapsed in current cycle (0-indexed)
    current_lr::T
end

function CosineAnnealingWarmRestarts(lr::T; T_0::Int, T_mult::Int=1, eta_min::T=T(1e-6)) where {T<:AbstractFloat}
    CosineAnnealingWarmRestarts{T}(lr, eta_min, T_0, T_mult, T_0, 0, lr)
end

function step!(sched::CosineAnnealingWarmRestarts{T}, epoch::Int, metric) where {T}
    sched.t_cur += 1
    if sched.t_cur >= sched.T_cur
        sched.t_cur = 0
        sched.T_cur = max(1, sched.T_cur * sched.T_mult)
        println("  LR scheduler: warm restart, new cycle length = $(sched.T_cur)")
    end
    sched.current_lr = sched.eta_min + T(0.5) * (sched.lr_max - sched.eta_min) * (1 + cos(T(π) * sched.t_cur / sched.T_cur))
    return sched.current_lr
end

# Adjusts the learning rate for the latent posterior subtrees (latent_mu,
# latent_log_sigma) independently of the global decoder LR. Called after every
# Optimisers.adjust! on the full tree to preserve the latent LR multiplier ratio.
function _adjust_latent_lr!(opt_state, latent_lr::Float32)
    for key in (:latent_mu, :latent_log_sigma)
        Optimisers.adjust!(getfield(opt_state, key), eta=latent_lr)
    end
end
