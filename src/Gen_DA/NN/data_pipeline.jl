module DataPipeline

using FFTW
using JLD2
using Random
import ..SpectralGrid, ..velocity_from_psi_hat

export load_trajectory, split_trajectory, make_sensor_array, make_per_sample_sensors,
       batch_partition, extract_observations, extract_observations_per_sample,
       extract_full_velocity, extract_vorticity_spectral,
       reduce_symmetries, reduce_trajectory

function load_trajectory(path::String)
    f = jldopen(path)
    data = NamedTuple{Tuple(Symbol.(keys(f)))}(Tuple(f[k] for k in keys(f)))
    close(f)
    return data
end

function split_trajectory(trajectory::AbstractArray{<:Real,3}, dt::Real, save_every::Int, T_train::Real)
    n_snaps = size(trajectory, 3)
    n_train = count(i -> (i - 1) * dt * save_every <= T_train, 1:n_snaps)
    return trajectory[:, :, 1:n_train], trajectory[:, :, n_train+1:end]
end

function make_sensor_array(N::Int, n_meas_space::Int, rng::AbstractRNG)
    all_ci = vec([CartesianIndex(i, j) for i in 1:N, j in 1:N])
    return shuffle(rng, all_ci)[1:n_meas_space]
end

# One random sensor layout per training snapshot, drawn once and held fixed.
# Returns sensor_lin_all: (n_meas_space, n_train) matrix of linearized indices (Int32).
function make_per_sample_sensors(N::Int, n_meas_space::Int, n_train::Int, rng::AbstractRNG)
    all_lin      = Int32.(1:N*N)
    sensor_lin   = zeros(Int32, n_meas_space, n_train)
    for i in 1:n_train
        sensor_lin[:, i] = shuffle(rng, all_lin)[1:n_meas_space]
    end
    return sensor_lin
end

function batch_partition(n_train::Int, batch_size::Int, rng::AbstractRNG)
    indices = shuffle(rng, collect(1:n_train))
    return [indices[i:min(i + batch_size - 1, n_train)] for i in 1:batch_size:n_train]
end

function _omega_to_psi_hat(omega::AbstractMatrix{Float32}, grid::SpectralGrid)
    return rfft(omega) ./ complex.(grid.lap)
end

function extract_observations(
    snaps::AbstractArray{Float32,3},
    indices::AbstractVector{Int},
    sensor_ci::AbstractVector{<:CartesianIndex{2}},
    grid::SpectralGrid,
)
    n_meas = length(sensor_ci)
    batch_size = length(indices)
    u_meas = zeros(Float32, n_meas, batch_size)
    v_meas = zeros(Float32, n_meas, batch_size)
    for (b, idx) in enumerate(indices)
        psi_hat = _omega_to_psi_hat(@view(snaps[:, :, idx]), grid)
        u, v = velocity_from_psi_hat(grid, psi_hat)
        for (m, ci) in enumerate(sensor_ci)
            u_meas[m, b] = u[ci]
            v_meas[m, b] = v[ci]
        end
    end
    return u_meas, v_meas
end

function extract_observations_per_sample(
    snaps::AbstractArray{Float32,3},
    indices::AbstractVector{Int},
    sensor_lin_all::AbstractMatrix{Int32},
    grid::SpectralGrid,
)
    n_meas     = size(sensor_lin_all, 1)
    batch_size = length(indices)
    u_meas = zeros(Float32, n_meas, batch_size)
    v_meas = zeros(Float32, n_meas, batch_size)
    for (b, idx) in enumerate(indices)
        psi_hat = _omega_to_psi_hat(@view(snaps[:, :, idx]), grid)
        u, v    = velocity_from_psi_hat(grid, psi_hat)
        u_flat  = vec(u)
        v_flat  = vec(v)
        for m in 1:n_meas
            u_meas[m, b] = u_flat[sensor_lin_all[m, idx]]
            v_meas[m, b] = v_flat[sensor_lin_all[m, idx]]
        end
    end
    return u_meas, v_meas
end

function extract_full_velocity(
    snaps::AbstractArray{Float32,3},
    indices::AbstractVector{Int},
    grid::SpectralGrid,
)
    N = grid.N
    batch_size = length(indices)
    u_full = zeros(Float32, N * N, batch_size)
    v_full = zeros(Float32, N * N, batch_size)
    for (b, idx) in enumerate(indices)
        psi_hat = _omega_to_psi_hat(@view(snaps[:, :, idx]), grid)
        u, v = velocity_from_psi_hat(grid, psi_hat)
        u_full[:, b] = vec(u)
        v_full[:, b] = vec(v)
    end
    return u_full, v_full
end

function extract_vorticity_spectral(
    snaps::AbstractArray{Float32,3},
    indices::AbstractVector{Int},
    nfreq::Int,
    NDOF::Int,
    N::Int,
)
    half = NDOF ÷ 2
    n_batch = length(indices)
    oh_re = zeros(Float32, nfreq, NDOF, n_batch)
    oh_im = zeros(Float32, nfreq, NDOF, n_batch)
    for (b, idx) in enumerate(indices)
        oh = rfft(@view snaps[:, :, idx])   # (N÷2+1, N)
        oh_re[:, 1:half, b]        .= real.(oh[1:nfreq, 1:half])
        oh_im[:, 1:half, b]        .= imag.(oh[1:nfreq, 1:half])
        oh_re[:, half+2:NDOF, b]   .= real.(oh[1:nfreq, N-half+1:N])
        oh_im[:, half+2:NDOF, b]   .= imag.(oh[1:nfreq, N-half+1:N])
        # column half+1 (Nyquist for NDOF) stays zero
    end
    return oh_re, oh_im
end

# Reduce all discrete and continuous symmetries of a single vorticity snapshot.
# Codebase convention: omega[y_idx, x_idx], rfft gives omega_hat[ky_idx, kx_idx].
# Discrete step 1 — shift-reflect: uses (ky=1, kx=0) = omega_hat[2, 1].
#   Correct group element: k = mod((n-1)*sector, 2n). For n=4: k = 3*sector mod 8.
#   Even k: pure y-shift. Odd k: x-flip + y-shift + negate (applied in physical space
#   to avoid the kx → -kx mixing that makes a spectral-only formula incorrect).
# Discrete step 2 — rotation: uses (ky=n, kx=0) = omega_hat[n+1, 1]; apply conj if
#   imag < 0 (complex conjugation = spatial rotation by π for a real field).
# Continuous — method of slices: aligns (ky=0, kx=1) = omega_hat[1, 2] to real axis.
function reduce_symmetries(ω::AbstractMatrix{Float32}, n::Int)
    N = size(ω, 1)
    ω_work = copy(ω)
    ω_hat  = rfft(ω_work)

    # Shift-reflect sector reduction
    θ = angle(ω_hat[2, 1])
    θ < 0 && (θ += 2π)
    sector = floor(Int, θ / (π / n))
    if sector != 0
        k           = mod((n - 1) * sector, 2n)
        shift_steps = mod(k * (N ÷ (2n)), N)
        if iseven(k)
            ω_work = circshift(ω_work, (shift_steps, 0))
        else
            ω_work = -circshift(ω_work[:, [1; N:-1:2]], (shift_steps, 0))
        end
        ω_hat = rfft(ω_work)
    end

    # Rotation reduction
    if imag(ω_hat[n + 1, 1]) < 0
        ω_hat = conj.(ω_hat)
    end

    # Method of slices: align (ky=0, kx=1) to positive real axis
    ϕ      = angle(ω_hat[1, 2])
    kx_int = reshape(Float32.(fftfreq(N) .* N), 1, N)
    ω_hat .*= exp.(complex.(zero(Float32), .-kx_int .* ϕ))

    return irfft(ω_hat, N)
end

function reduce_trajectory(snaps::Array{Float32,3}, n::Int)
    out = similar(snaps)
    for t in axes(snaps, 3)
        out[:, :, t] = reduce_symmetries(@view(snaps[:, :, t]), n)
    end
    return out
end

end # module DataPipeline
