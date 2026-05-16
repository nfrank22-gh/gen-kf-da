import ..Solver: KfRhs, implicit_term, implicit_solve
import Reactant
using AbstractFFTs
using Statistics

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

# ---------------------------------------------------------------------------
# Adjoint (backward-pass) functions for the KF solver chain.
# These are called by the custom phase-2 training loop; Enzyme never sees
# the solver internals.
# ---------------------------------------------------------------------------

# Adjoint of _explicit_term_batched w.r.t. omega_hat.
# Re-runs the forward to get physical-space intermediates, then propagates
# d_g (adjoint of the return value) back to d_omega_hat.
#
# Key sign rules: dxOp = i*kx and dyOp = i*ky are purely imaginary, so
#   conj(dxOp) = -dxOp  and  conj(dyOp) = -dyOp.
# laplacianOp and diffOp are real (Laplacian of real wavenumbers).
# M (dealiasing mask) is real.
# Adjoint of rfft is irfft; adjoint of irfft is rfft (standard L2 result).
# Adjoint of pointwise multiply by A is multiply by conj(A).
function _explicit_term_batched_adjoint(rhs::KfRhs, omega_hat, d_g)
    N = size(rhs.M, 2)

    # Re-run forward to recover physical-space fields needed for the adjoint.
    psi_hat = omega_hat ./ rhs.laplacianOp
    u_hat   =  rhs.dyOp .* psi_hat
    v_hat   = .-rhs.dxOp .* psi_hat
    u       = irfft(u_hat,              N, 1:2)
    v       = irfft(v_hat,              N, 1:2)
    dw_dx   = irfft(rhs.dxOp .* omega_hat, N, 1:2)
    dw_dy   = irfft(rhs.dyOp .* omega_hat, N, 1:2)

    # (L9) adjoint: g = adv_hat .* M + forcingHat  (M real)
    d_adv_hat = d_g .* rhs.M

    # (L8) adjoint: adv_hat = rfft(-(u.*dw_dx + v.*dw_dy))
    d_tmp   = irfft(d_adv_hat, N, 1:2)   # adjoint of rfft is irfft
    d_u     = .-d_tmp .* dw_dx
    d_v     = .-d_tmp .* dw_dy
    d_dw_dx = .-d_tmp .* u
    d_dw_dy = .-d_tmp .* v

    # (L4,L5) adjoint: u = irfft(u_hat), v = irfft(v_hat)
    d_u_hat = rfft(d_u, 1:2)
    d_v_hat = rfft(d_v, 1:2)

    # (L2) adjoint: u_hat = dyOp .* psi_hat  →  conj(dyOp) = -dyOp
    # (L3) adjoint: v_hat = -dxOp .* psi_hat  →  conj(-dxOp) = dxOp
    d_psi_hat = conj.(rhs.dyOp) .* d_u_hat .+ conj.(.-rhs.dxOp) .* d_v_hat

    # (L1) adjoint: psi_hat = omega_hat / laplacianOp  (real denominator)
    d_omega_hat = d_psi_hat ./ rhs.laplacianOp

    # (L6,L7) adjoint: dxOp.*omega_hat, dyOp.*omega_hat  →  conj(dxOp)=-dxOp, conj(dyOp)=-dyOp
    d_omega_hat = d_omega_hat .+
                  conj.(rhs.dxOp) .* rfft(d_dw_dx, 1:2) .+
                  conj.(rhs.dyOp) .* rfft(d_dw_dy, 1:2)

    return d_omega_hat
end

# Adjoint of _kf_step_batched w.r.t. omega_hat_in.
# Re-runs the 5-stage IMEX RK forward to recover all intermediate states,
# then propagates d_u_out (adjoint of the step output) back to d_omega_hat_in.
function _kf_step_batched_adjoint(rhs::KfRhs, omega_hat_in, dt, d_u_out)
    # ── Forward re-run (store u and h at each of the 6 stage boundaries) ──
    u = omega_hat_in
    h = zero(omega_hat_in)
    u_stages = Vector{typeof(omega_hat_in)}(undef, 6)   # u_stages[i] = u entering stage i
    h_stages = Vector{typeof(omega_hat_in)}(undef, 6)   # h_stages[i] = h entering stage i
    u_stages[1] = u
    h_stages[1] = h
    for i in 1:5
        g       = _explicit_term_batched(rhs, u)
        h       = g .+ _relax_beta[i] .* h
        mu      = 0.5f0 * dt * (_relax_alpha[i+1] - _relax_alpha[i])
        imp_rhs = u .+ _relax_gamma[i] .* dt .* h .+ mu .* implicit_term(rhs, u)
        u       = implicit_solve(rhs, imp_rhs, mu)
        u_stages[i+1] = u
        h_stages[i+1] = h
    end

    # ── Backward through 5 stages (i = 5 … 1) ──
    # d_u     = adjoint of u_i (the current stage output)
    # d_h_ext = adjoint of h_{i-1} arriving from stage i's backward
    d_u     = d_u_out
    d_h_ext = zero(omega_hat_in)

    for i in 5:-1:1
        mu = 0.5f0 * dt * (_relax_alpha[i+1] - _relax_alpha[i])

        # u_i = implicit_solve: imp_rhs / (1 - mu*diffOp).  diffOp real → same denominator.
        d_imp_rhs = d_u ./ (1f0 .- mu .* rhs.diffOp)

        # imp_rhs = u_{i-1}*(1 + mu*diffOp) + gamma[i]*dt * h_i
        d_u_prev_imp = (1f0 .+ mu .* rhs.diffOp) .* d_imp_rhs
        d_h_i_imp    = _relax_gamma[i] * dt .* d_imp_rhs

        # Total adjoint of h_i (from stage i+1 + from imp_rhs of this stage)
        d_h_i = d_h_ext .+ d_h_i_imp

        # h_i = g_i + beta[i]*h_{i-1}
        d_g_i   = d_h_i
        d_h_ext = _relax_beta[i] .* d_h_i   # becomes d_h_{i-1} for next iteration

        # g_i = _explicit_term_batched(rhs, u_{i-1})
        d_u_prev_g = _explicit_term_batched_adjoint(rhs, u_stages[i], d_g_i)

        d_u = d_u_prev_imp .+ d_u_prev_g
    end

    return d_u   # = d_omega_hat_in
end

"""
    relax_and_store(rhs, omega_hat_0, n_steps, dt)

Forward-only pass: run `n_steps` KF solver steps, storing omega_hat before each
step.  Returns a 4-D array of shape `(N÷2+1, N, batch, n_steps+1)` where slice k
(1-indexed) is the spectral field that enters solver step k.

Compiled with `Reactant.@compile`; the loop is unrolled at trace time (n_steps is
a concrete Int), so Enzyme is never asked to differentiate through it.
"""
function relax_and_store(rhs::KfRhs, omega_hat_0, n_steps::Int, dt::Float32)
    N_freq, N_x, B = size(omega_hat_0)
    traj      = reshape(omega_hat_0, N_freq, N_x, B, 1)
    omega_hat = omega_hat_0
    for _ in 1:n_steps
        omega_hat = _kf_step_batched(rhs, omega_hat, dt)
        traj = cat(traj, reshape(omega_hat, N_freq, N_x, B, 1); dims=4)
    end
    return traj   # traj[:,:,:,end] is the final relaxed state
end

"""
    relax_adj_full(rhs, traj, n_steps, dt, d_omega_relaxed, N_out)

Backward pass for the full relaxation chain
    omega_gen → rfft → omega_hat_0 → relax → omega_hat_n → irfft → omega_relaxed.

Given `d_omega_relaxed` (real, gradient of the loss w.r.t. omega_relaxed), returns
`d_omega_gen` (real, gradient w.r.t. the upsampler output).

`traj` is the trajectory from `relax_and_store`; slice `traj[:,:,:,k]` is the
spectral field entering solver step k.
"""
function relax_adj_full(rhs::KfRhs, traj, n_steps::Int, dt::Float32,
                        d_omega_relaxed, N_out::Int)
    # adjoint of irfft is rfft
    d_omega_hat = rfft(d_omega_relaxed, 1:2)

    # adjoint of n_steps KF steps, stepping backward through the stored trajectory
    for i in n_steps:-1:1
        d_omega_hat = _kf_step_batched_adjoint(rhs, traj[:, :, :, i], dt, d_omega_hat)
    end

    # adjoint of rfft is irfft
    return irfft(d_omega_hat, N_out, 1:2)
end

"""
    RelaxedDecoder(decoder, rhs, n_steps, dt)

Callable struct composing a Lux decoder with KF solver relaxation.  Calling
`rd(N, x, ps, st)` runs the full chain:

    latent x  →  eval_decoder_vort  →  rfft  →  relax  →  irfft  →  omega_relaxed

Returns `(omega_relaxed, new_st)` with the same signature as `eval_decoder_vort`.
Works on both CPU arrays (for inference / plotting) and Reactant arrays (for
compiled GPU passes).
"""
struct RelaxedDecoder{M}
    decoder::M
    rhs::KfRhs
    n_steps::Int
    dt::Float32
end

function (rd::RelaxedDecoder)(N::Int, x, ps, st)
    omega_gen, new_st = eval_decoder_vort(rd.decoder, N, x, ps, st)
    omega_hat         = rfft(omega_gen, 1:2)
    omega_hat_r       = relax(rd.rhs, omega_hat, rd.n_steps, rd.dt)
    return irfft(omega_hat_r, N, 1:2), new_st
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
