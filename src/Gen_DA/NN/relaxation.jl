import ..Solver: KfRhs, implicit_term, implicit_solve
import Reactant
using AbstractFFTs

const _relax_alpha = Float32[0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
const _relax_beta  = Float32[0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
const _relax_gamma = Float32[0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

# Explicit term for a batched omega_hat of shape (N÷2+1, N, batch).
# Identical to Solver.explicit_term but uses irfft/rfft with explicit dims
# so that the batch dimension is not transformed.
function _explicit_term_batched(rhs::KfRhs, omega_hat)
    N       = size(rhs.M, 2)
    psi_hat = omega_hat ./ rhs.laplacianOp
    u_hat   =  rhs.dyOp .* psi_hat
    v_hat   = .-rhs.dxOp .* psi_hat
    u     = irfft(u_hat, N, 1:2)
    v     = irfft(v_hat, N, 1:2)
    dw_dx = irfft(rhs.dxOp .* omega_hat, N, 1:2)
    dw_dy = irfft(rhs.dyOp .* omega_hat, N, 1:2)
    adv_hat = rfft(.-(u .* dw_dx .+ v .* dw_dy), 1:2)
    return adv_hat .* rhs.M .+ rhs.forcingHat
end

function _kf_step_batched(rhs::KfRhs, omega_hat, dt)
    u = omega_hat
    h = zero(omega_hat)
    for i in 1:5
        g       = _explicit_term_batched(rhs, u)
        h       = g .+ _relax_beta[i] .* h
        mu      = 0.5f0 * dt * (_relax_alpha[i+1] - _relax_alpha[i])
        imp_rhs = u .+ _relax_gamma[i] .* dt .* h .+ mu .* implicit_term(rhs, u)
        u       = implicit_solve(rhs, imp_rhs, mu)
    end
    return u
end

"""
    relax(rhs, omega_hat, n_steps, dt)

Advance a batch of spectral vorticity fields `omega_hat` (shape `(N÷2+1, N, batch)`)
forward by `n_steps` KF solver steps of size `dt`.  Returns the relaxed
`omega_hat` with the same shape.
"""
function relax(rhs::KfRhs, omega_hat, n_steps::Int, dt::Float32)
    # @trace promotes the loop bounds to traced values inside @compile so the loop
    # is lowered to a single stablehlo.while op instead of being unrolled into the
    # MLIR graph — necessary for large n_steps where unrolling causes compilation to hang.
    Reactant.@trace for _ in 1:n_steps
        omega_hat = _kf_step_batched(rhs, omega_hat, dt)
    end
    return omega_hat
end

"""
    loss_fn_relaxation(model, N_out, x, ps, st, rhs, n_steps, dt_relax, omega_trg, thetas)

Phase-2 loss: decode latents → rfft → relax by `n_steps` solver steps → irfft → SWD
against `omega_trg`.  The SWD is computed in physical space on the relaxed vorticity.
"""
function loss_fn_relaxation(model, N_out::Integer, x, ps, st,
                             rhs::KfRhs, n_steps::Int, dt_relax::Float32,
                             omega_trg, thetas)
    omega, new_st        = eval_decoder_vort(model, N_out, x, ps, st)
    omega_hat            = rfft(omega, 1:2)
    omega_hat_relaxed    = relax(rhs, omega_hat, n_steps, dt_relax)
    omega_relaxed        = irfft(omega_hat_relaxed, N_out, 1:2)
    flat_dim = N_out * N_out
    P = reshape(omega_relaxed, flat_dim, size(omega_relaxed, 3))
    Q = reshape(omega_trg,     flat_dim, size(omega_trg,     3))
    return sliced_wasserstein(P, Q, thetas), new_st
end
