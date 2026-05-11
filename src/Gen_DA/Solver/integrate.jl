using Random

function random_ic(N; T=Float32, seed=42)
    rng = MersenneTwister(seed)
    omega_hat = randn(rng, Complex{T}, N÷2+1, N)
    KX, KY = get_K(N; T=T)
    K = sqrt.(KX.^2 .+ KY.^2)
    M = get_dealias_mask(N; T=T)
    @. omega_hat *= K * exp(-K / T(4)) * M
    return omega_hat  # CPU array; caller converts to device
end

function random_particles_ic(npart; T=Float32, seed=123)
    rng = MersenneTwister(seed)
    xp = T.(rand(rng, npart) .* 2pi)
    yp = T.(rand(rng, npart) .* 2pi)
    return xp, yp  # CPU arrays; caller converts to device
end

function integrate(step_fn, rhs::KfRhs, omega_hat, dt, n_steps; save_every=1)
    N = size(rhs.M, 2)
    n_saves = n_steps ÷ save_every
    trajectory = zeros(Float32, N, N, n_saves)
    save_idx = 1
    for i in 1:n_steps
        omega_hat = step_fn(rhs, omega_hat, dt)
        if i % save_every == 0
            trajectory[:, :, save_idx] .= irfft(Array(omega_hat), N)
            save_idx += 1
        end
    end
    return trajectory, omega_hat
end

function integrate(step_fn, rhs::KfRhs, omega_hat, xp, yp, dt, n_steps; save_every=1)
    N     = size(rhs.M, 2)
    npart = length(xp)
    n_saves = n_steps ÷ save_every
    trajectory = zeros(Float32, N, N, n_saves)
    xp_traj    = zeros(Float32, npart, n_saves)
    yp_traj    = zeros(Float32, npart, n_saves)
    save_idx = 1
    for i in 1:n_steps
        omega_hat, xp, yp = step_fn(rhs, omega_hat, xp, yp, dt)
        if i % save_every == 0
            trajectory[:, :, save_idx] .= irfft(Array(omega_hat), N)
            xp_traj[:, save_idx]       .= Array(xp)
            yp_traj[:, save_idx]       .= Array(yp)
            save_idx += 1
        end
    end
    return trajectory, omega_hat, xp_traj, yp_traj
end
