using Optimisers

export build_optimizer, ReduceOnPlateau, step!

function build_optimizer(lr::AbstractFloat)
    Adam(lr)
end

mutable struct ReduceOnPlateau{T<:AbstractFloat}
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

function step!(sched::ReduceOnPlateau, metric)
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
