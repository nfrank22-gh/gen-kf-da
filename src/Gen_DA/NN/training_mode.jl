import Reactant

# ---------------------------------------------------------------------------
# Training modes — each owns the three data closures, thetas_flat_dim,
# and any mode-specific state.  Construct the mode explicitly, then pass
# it to TrainingSession.  train! dispatches on the concrete type.
#
# kl_weight is passed dynamically via the data tuple so that
# KL warmup annealing takes effect without recompiling the XLA graph.
# ---------------------------------------------------------------------------

struct VorticityMode
    batch_size::Int
    n_full::Int
    thetas_flat_dim::Int
    freeze_upsampler::Bool
    prepare_epoch::Any    # full_batches -> (omega_cpu,)
    get_data_batch::Any   # (data_all, cols) -> (omega_ra,)
    train_loss::Any       # (model, ps, st, data) -> (loss, st, ())
end

struct ObservationsMode
    batch_size::Int
    n_full::Int
    thetas_flat_dim::Int
    freeze_upsampler::Bool
    prepare_epoch::Any    # full_batches -> (u_cpu, v_cpu)
    get_data_batch::Any   # (data_all, cols) -> (u_ra, v_ra)
    train_loss::Any       # (model, ps, st, data) -> (loss, st, ())
end


function VorticityMode(train_snaps::Array{Float32,3}, N::Int, batch_size::Int;
                       freeze_upsampler::Bool=false,
                       h2_weight::Float32=0f0,
                       recon_loss::Symbol=:swd,
                       sinkhorn_eps::Float32=0.1f0,
                       sinkhorn_n_iter::Int=100,
                       sinkhorn_metric::Symbol=:sq_l2,
                       mahalanobis_L::Union{Nothing,Matrix{Float32}}=nothing,
                       mahalanobis_rank::Union{Nothing,Int}=nothing,
                       rng=nothing)
    n_full = size(train_snaps, 3) ÷ batch_size
    if sinkhorn_metric == :rand_lr_mahalanobis
        mahalanobis_rank === nothing && error("mahalanobis_rank required for :rand_lr_mahalanobis")
        r = mahalanobis_rank
        prepare_epoch_fn = function(full_batches)
            omega_cpu = zeros(Float32, N, N, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                slabs = (i-1)*batch_size+1 : i*batch_size
                omega_cpu[:, :, slabs] = train_snaps[:, :, batch_indices]
            end
            L = rng !== nothing ? randn(rng, Float32, N * N, r) : randn(Float32, N * N, r)
            return (omega_cpu, L)
        end
        get_data_batch_fn = function(data_all, cols)
            omega_all, L = data_all
            return (Reactant.to_rarray(omega_all[:, :, cols]), Reactant.to_rarray(L))
        end
        train_loss_fn = function(m, params, states, data)
            omega_trg, L_ra, thetas, cols, eps, kl_w = data
            loss_fn_vort_state(m, N, params, states, omega_trg, thetas, cols,
                               eps, sum(kl_w);
                               h2_weight=h2_weight,
                               recon_loss=recon_loss,
                               sinkhorn_eps=sinkhorn_eps,
                               sinkhorn_n_iter=sinkhorn_n_iter,
                               sinkhorn_metric=:rand_lr_mahalanobis,
                               mahalanobis_L=L_ra)
        end
    else
        L_device = mahalanobis_L !== nothing ? Reactant.to_rarray(mahalanobis_L) : nothing
        prepare_epoch_fn = function(full_batches)
            omega_cpu = zeros(Float32, N, N, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                slabs = (i-1)*batch_size+1 : i*batch_size
                omega_cpu[:, :, slabs] = train_snaps[:, :, batch_indices]
            end
            return (omega_cpu,)
        end
        get_data_batch_fn = function(data_all, cols)
            omega_all, = data_all
            return (Reactant.to_rarray(omega_all[:, :, cols]),)
        end
        train_loss_fn = function(m, params, states, data)
            omega_trg, thetas, cols, eps, kl_w = data
            loss_fn_vort_state(m, N, params, states, omega_trg, thetas, cols,
                               eps, sum(kl_w);
                               h2_weight=h2_weight,
                               recon_loss=recon_loss,
                               sinkhorn_eps=sinkhorn_eps,
                               sinkhorn_n_iter=sinkhorn_n_iter,
                               sinkhorn_metric=sinkhorn_metric,
                               mahalanobis_L=L_device)
        end
    end
    thetas_flat_dim = recon_loss == :sinkhorn ? 1 : N * N
    VorticityMode(batch_size, n_full, thetas_flat_dim, freeze_upsampler,
                  prepare_epoch_fn, get_data_batch_fn, train_loss_fn)
end

# Shared factory for the two observation modes that use a single fixed sensor_lin vector.
# extractor_fn: (snaps, batch_indices, grid) -> (u_cpu, v_cpu) for n_meas rows.
function _make_shared_sensor_mode(
    train_snaps::Array{Float32,3}, N::Int, batch_size::Int,
    n_meas::Int, sensor_lin,
    extractor_fn,
    freeze_upsampler::Bool, recon_loss::Symbol,
    thetas_flat_dim::Int,
    sinkhorn_eps::Float32, sinkhorn_n_iter::Int,
    sinkhorn_metric::Symbol=:sq_l2,
    mahalanobis_L::Union{Nothing,Matrix{Float32}}=nothing,
    mahalanobis_rank::Union{Nothing,Int}=nothing,
    rng=nothing,
    h2_weight::Float32=0f0,
)
    n_full    = size(train_snaps, 3) ÷ batch_size
    data_grid = SpectralGrid(N)
    if sinkhorn_metric == :rand_lr_mahalanobis
        mahalanobis_rank === nothing && error("mahalanobis_rank required for :rand_lr_mahalanobis")
        r = mahalanobis_rank
        prepare_epoch_fn = function(full_batches)
            u_cpu = zeros(Float32, n_meas, batch_size * n_full)
            v_cpu = zeros(Float32, n_meas, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                cols = (i-1)*batch_size+1 : i*batch_size
                u_cpu[:, cols], v_cpu[:, cols] = extractor_fn(train_snaps, batch_indices, data_grid)
            end
            L = rng !== nothing ? randn(rng, Float32, 2 * n_meas, r) : randn(Float32, 2 * n_meas, r)
            return u_cpu, v_cpu, L
        end
        get_data_batch_fn = function(data_all, cols)
            u_all, v_all, L = data_all
            return (Reactant.to_rarray(u_all[:, cols]), Reactant.to_rarray(v_all[:, cols]),
                    Reactant.to_rarray(L))
        end
        train_loss_fn = function(m, params, states, data)
            u_trg, v_trg, L_ra, thetas, cols, eps, kl_w = data
            loss_fn(m, N, params, states, u_trg, v_trg, sensor_lin, thetas, cols,
                    eps, sum(kl_w);
                    h2_weight=h2_weight,
                    recon_loss=recon_loss,
                    sinkhorn_eps=sinkhorn_eps,
                    sinkhorn_n_iter=sinkhorn_n_iter,
                    sinkhorn_metric=:rand_lr_mahalanobis,
                    mahalanobis_L=L_ra)
        end
    else
        L_device  = mahalanobis_L !== nothing ? Reactant.to_rarray(mahalanobis_L) : nothing
        prepare_epoch_fn = function(full_batches)
            u_cpu = zeros(Float32, n_meas, batch_size * n_full)
            v_cpu = zeros(Float32, n_meas, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                cols = (i-1)*batch_size+1 : i*batch_size
                u_cpu[:, cols], v_cpu[:, cols] = extractor_fn(train_snaps, batch_indices, data_grid)
            end
            return u_cpu, v_cpu
        end
        get_data_batch_fn = function(data_all, cols)
            u_all, v_all = data_all
            return (Reactant.to_rarray(u_all[:, cols]), Reactant.to_rarray(v_all[:, cols]))
        end
        train_loss_fn = function(m, params, states, data)
            u_trg, v_trg, thetas, cols, eps, kl_w = data
            loss_fn(m, N, params, states, u_trg, v_trg, sensor_lin, thetas, cols,
                    eps, sum(kl_w);
                    h2_weight=h2_weight,
                    recon_loss=recon_loss,
                    sinkhorn_eps=sinkhorn_eps,
                    sinkhorn_n_iter=sinkhorn_n_iter,
                    sinkhorn_metric=sinkhorn_metric,
                    mahalanobis_L=L_device)
        end
    end

    return ObservationsMode(batch_size, n_full, thetas_flat_dim, freeze_upsampler,
                            prepare_epoch_fn, get_data_batch_fn, train_loss_fn)
end

function ObservationsMode(train_snaps::Array{Float32,3}, N::Int, batch_size::Int,
                          n_meas_space, sensor_ci;
                          freeze_upsampler::Bool=false,
                          h2_weight::Float32=0f0,
                          random_sensors::Bool=false, rng=nothing,
                          recon_loss::Symbol=:swd,
                          sinkhorn_eps::Float32=0.1f0,
                          sinkhorn_n_iter::Int=100,
                          sinkhorn_metric::Symbol=:sq_l2,
                          mahalanobis_L::Union{Nothing,Matrix{Float32}}=nothing,
                          mahalanobis_rank::Union{Nothing,Int}=nothing)
    if isinf(n_meas_space)
        extractor  = (snaps, idxs, grid) -> DataPipeline.extract_full_velocity(snaps, idxs, grid)
        sensor_lin = LinearIndices((N, N))[:]
        td = recon_loss == :sinkhorn ? 1 : 2 * N * N
        return _make_shared_sensor_mode(train_snaps, N, batch_size, N * N, sensor_lin,
                                        extractor, freeze_upsampler,
                                        recon_loss, td,
                                        sinkhorn_eps, sinkhorn_n_iter, sinkhorn_metric,
                                        mahalanobis_L, mahalanobis_rank, rng, h2_weight)
    end

    if random_sensors
        (mahalanobis_L !== nothing || sinkhorn_metric in (:lr_mahalanobis, :rand_lr_mahalanobis)) && error(
            "sinkhorn_metric = :lr_mahalanobis / :rand_lr_mahalanobis is not supported with random_sensors = true")
        n_train        = size(train_snaps, 3)
        n_full         = n_train ÷ batch_size
        data_grid      = SpectralGrid(N)
        sensor_lin_all = DataPipeline.make_per_sample_sensors(N, n_meas_space, n_train, rng)
        prepare_epoch_fn = function(full_batches)
            u_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
            v_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
            for (i, batch_indices) in enumerate(full_batches)
                cols = (i-1)*batch_size+1 : i*batch_size
                u_cpu[:, cols], v_cpu[:, cols] =
                    DataPipeline.extract_observations_per_sample(
                        train_snaps, batch_indices, sensor_lin_all, data_grid)
            end
            return u_cpu, v_cpu
        end

        get_data_batch_fn = function(data_all, cols)
            u_all, v_all = data_all
            return (Reactant.to_rarray(u_all[:, cols]),
                    Reactant.to_rarray(v_all[:, cols]),
                    Reactant.to_rarray(sensor_lin_all[:, cols]))
        end
        train_loss_fn = function(m, params, states, data)
            u_trg, v_trg, sensor_lin_batch, thetas, cols, eps, kl_w = data
            loss_fn_per_sample_sensors(
                m, N, params, states, u_trg, v_trg, sensor_lin_batch, thetas, cols,
                eps, sum(kl_w);
                h2_weight=h2_weight,
                recon_loss=recon_loss,
                sinkhorn_eps=sinkhorn_eps,
                sinkhorn_n_iter=sinkhorn_n_iter,
                sinkhorn_metric=sinkhorn_metric)
        end

        td = recon_loss == :sinkhorn ? 1 : 2 * n_meas_space
        return ObservationsMode(batch_size, n_full, td, freeze_upsampler,
                                prepare_epoch_fn, get_data_batch_fn, train_loss_fn)
    end

    extractor  = (snaps, idxs, grid) -> DataPipeline.extract_observations(snaps, idxs, sensor_ci, grid)
    sensor_lin = LinearIndices((N, N))[sensor_ci]
    td = recon_loss == :sinkhorn ? 1 : 2 * n_meas_space
    return _make_shared_sensor_mode(train_snaps, N, batch_size, n_meas_space, sensor_lin,
                                    extractor, freeze_upsampler,
                                    recon_loss, td,
                                    sinkhorn_eps, sinkhorn_n_iter, sinkhorn_metric,
                                    mahalanobis_L, mahalanobis_rank, rng, h2_weight)
end
