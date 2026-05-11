using Gen_DA.NN
using Gen_DA.NN.DataPipeline
using Reactant, Random

Reactant.set_default_backend("cuda")
const dev = reactant_device()

function main()
    T            = Float32
    Re           = 100
    N            = 128
    npart        = 40
    T_train      = 800.0
    n_meas_space = 20
    batch_size   = 100
    n_epochs     = 10
    num_freq     = 16
    latent_dim   = 100
    layers       = [latent_dim, 512, 1024]
    n_slices     = 50

    rng = Xoshiro(123)

    traj = load_trajectory("data/particles/Re$(Re)_N$(N)_npart$(npart)/trajectory.jld2")
    train_snaps, _ = split_trajectory(traj.trajectory, traj.dt, traj.save_every, T_train)
    n_train = size(train_snaps, 3)
    println("Training on $n_train snapshots (T_train=$T_train)")

    sensor_ci  = make_sensor_array(N, n_meas_space, rng)
    sensor_lin = LinearIndices((N, N))[sensor_ci]

    model, ps, st = VortFourierDecoder(layers, num_freq, T(2π), rng, T)
    ps     = ps |> dev
    st     = st |> dev
    thetas = randn(rng, T, n_slices, 2 * n_meas_space) |> dev

    # Compile once for the fixed batch size; undersized last batch is skipped each epoch
    x_dummy   = Reactant.to_rarray(zeros(T, latent_dim, batch_size))
    u_dummy   = Reactant.to_rarray(zeros(T, n_meas_space, batch_size))
    v_dummy   = Reactant.to_rarray(zeros(T, n_meas_space, batch_size))
    loss_compiled = @compile loss_fn(model, N, x_dummy, ps, st, u_dummy, v_dummy, sensor_lin, thetas)
    println("Compiled loss function.")

    for epoch in 1:n_epochs
        batches    = batch_partition(n_train, batch_size, rng)
        epoch_loss = zero(T)
        n_full     = 0
        for batch_indices in batches
            length(batch_indices) == batch_size || continue
            u_trg, v_trg = extract_observations(train_snaps, batch_indices, sensor_ci, N)
            x = Reactant.to_rarray(randn(rng, T, latent_dim, batch_size))
            l, st = loss_compiled(model, N, x, Reactant.to_rarray(u_trg), Reactant.to_rarray(v_trg),
                                  sensor_lin, thetas)
            epoch_loss += Array(l)[]
            n_full += 1
        end
        println("epoch $epoch  loss = $(epoch_loss / max(n_full, 1))")
    end
end

main()
