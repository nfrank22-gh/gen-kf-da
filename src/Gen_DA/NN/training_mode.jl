import Reactant

# ---------------------------------------------------------------------------
# Training modes — each owns the three data closures, thetas_flat_dim,
# and any mode-specific state.  Construct the mode explicitly, then pass
# it to TrainingSession.  train! dispatches on the concrete type.
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
                       freeze_upsampler::Bool=false, kl_weight::Float32=0f0)
    n_full = size(train_snaps, 3) ÷ batch_size
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
        omega_trg, thetas, cols, eps = data
        l, new_st = loss_fn_vort_state(m, N, params, states, omega_trg, thetas, cols, eps, kl_weight)
        l, new_st, (;)
    end
    VorticityMode(batch_size, n_full, N * N, freeze_upsampler,
                  prepare_epoch_fn, get_data_batch_fn, train_loss_fn)
end

function ObservationsMode(train_snaps::Array{Float32,3}, N::Int, batch_size::Int,
                          n_meas_space::Int, sensor_ci;
                          freeze_upsampler::Bool=false, kl_weight::Float32=0f0)
    n_full     = size(train_snaps, 3) ÷ batch_size
    data_grid  = SpectralGrid(N)
    sensor_lin = LinearIndices((N, N))[sensor_ci]
    prepare_epoch_fn = function(full_batches)
        u_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
        v_cpu = zeros(Float32, n_meas_space, batch_size * n_full)
        for (i, batch_indices) in enumerate(full_batches)
            cols = (i-1)*batch_size+1 : i*batch_size
            u_cpu[:, cols], v_cpu[:, cols] =
                DataPipeline.extract_observations(train_snaps, batch_indices, sensor_ci, data_grid)
        end
        return u_cpu, v_cpu
    end
    get_data_batch_fn = function(data_all, cols)
        u_all, v_all = data_all
        return (Reactant.to_rarray(u_all[:, cols]), Reactant.to_rarray(v_all[:, cols]))
    end
    train_loss_fn = function(m, params, states, data)
        u_trg, v_trg, thetas, cols, eps = data
        l, new_st = loss_fn(m, N, params, states, u_trg, v_trg, sensor_lin, thetas, cols, eps, kl_weight)
        l, new_st, (;)
    end
    ObservationsMode(batch_size, n_full, 2 * n_meas_space, freeze_upsampler,
                     prepare_epoch_fn, get_data_batch_fn, train_loss_fn)
end
