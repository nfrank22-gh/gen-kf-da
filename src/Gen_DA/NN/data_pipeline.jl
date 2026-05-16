module DataPipeline

using FFTW
using JLD2
using Random
import ..SpectralGrid, ..velocity_from_psi_hat

export load_trajectory, split_trajectory, make_sensor_array, batch_partition, extract_observations, extract_vorticity_spectral

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

function batch_partition(n_train::Int, batch_size::Int, rng::AbstractRNG)
    indices = shuffle(rng, collect(1:n_train))
    return [indices[i:min(i + batch_size - 1, n_train)] for i in 1:batch_size:n_train]
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
        omega_hat = rfft(@view snaps[:, :, idx])
        psi_hat   = omega_hat ./ complex.(grid.lap)
        u, v = velocity_from_psi_hat(grid, psi_hat)
        for (m, ci) in enumerate(sensor_ci)
            u_meas[m, b] = u[ci]
            v_meas[m, b] = v[ci]
        end
    end
    return u_meas, v_meas
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

end # module DataPipeline
