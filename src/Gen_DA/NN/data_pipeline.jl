module DataPipeline

using FFTW
using JLD2
using Random

export load_trajectory, split_trajectory, make_sensor_array, batch_partition, extract_observations

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

const _L = Float32(2π)

function _spectral_ops(N::Int)
    dx = _L / N
    ky_1d = Float32.(2π .* rfftfreq(N, 1/dx))
    kx_1d = Float32.(2π .* fftfreq(N, 1/dx))
    KY = reshape(ky_1d, N÷2+1, 1)
    KX = reshape(kx_1d, 1, N)
    lap = -(KX.^2 .+ KY.^2)
    lap[1, 1] = 1f0
    return KX, KY, lap
end

function extract_observations(
    snaps::AbstractArray{Float32,3},
    indices::AbstractVector{Int},
    sensor_ci::AbstractVector{<:CartesianIndex{2}},
    N::Int,
)
    KX, KY, lap = _spectral_ops(N)
    n_meas = length(sensor_ci)
    batch_size = length(indices)
    u_meas = zeros(Float32, n_meas, batch_size)
    v_meas = zeros(Float32, n_meas, batch_size)
    for (b, idx) in enumerate(indices)
        omega_hat = rfft(@view snaps[:, :, idx])
        psi_hat   = omega_hat ./ complex.(lap)
        u = irfft(complex.(zero(KY), KY) .* psi_hat, N)
        v = irfft(complex.(zero(KX), .-KX) .* psi_hat, N)
        for (m, ci) in enumerate(sensor_ci)
            u_meas[m, b] = u[ci]
            v_meas[m, b] = v[ci]
        end
    end
    return u_meas, v_meas
end

end # module DataPipeline
