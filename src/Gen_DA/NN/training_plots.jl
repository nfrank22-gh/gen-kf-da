module TrainingPlots

using CairoMakie
using FFTW
using Statistics

export plot_train_loss_curve, plot_eval_swd_curve, plot_vorticity_panel, plot_energy_spectrum

function plot_train_loss_curve(train_losses::AbstractVector, model_dir::AbstractString)
    mkpath(model_dir)
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel="Epoch", ylabel="Loss", title="Training Loss")
    lines!(ax, eachindex(train_losses), train_losses)
    save(joinpath(model_dir, "loss_curve.png"), fig)
end

function plot_eval_swd_curve(eval_swds::AbstractVector, eval_epochs::AbstractVector{<:Integer},
                             model_dir::AbstractString)
    mkpath(model_dir)
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel="Epoch", ylabel="Eval SWD", title="Eval Sliced Wasserstein Distance")
    lines!(ax,   eval_epochs, eval_swds)
    scatter!(ax, eval_epochs, eval_swds)
    save(joinpath(model_dir, "eval_swd_curve.png"), fig)
end

function plot_vorticity_panel(gen_omega::AbstractArray{<:Real, 3}, gt_omega::AbstractArray{<:Real, 3},
                              model_dir::AbstractString; n_samples::Int=4)
    mkpath(model_dir)
    n = min(n_samples, size(gen_omega, 3), size(gt_omega, 3))
    fig = Figure(size=(n * 200, 420))
    for i in 1:n
        ax1 = Axis(fig[1, i]; title="Generated $i", xlabel="x", ylabel="y", aspect=DataAspect())
        ax2 = Axis(fig[2, i]; title="Ground truth $i", xlabel="x", ylabel="y", aspect=DataAspect())
        lim1 = max(maximum(abs, gen_omega[:, :, i]), eps(Float32))
        lim2 = max(maximum(abs, gt_omega[:, :, i]),  eps(Float32))
        heatmap!(ax1, gen_omega[:, :, i]; colormap=:RdBu, colorrange=(-lim1, lim1))
        heatmap!(ax2, gt_omega[:, :, i];  colormap=:RdBu, colorrange=(-lim2, lim2))
    end
    save(joinpath(model_dir, "vorticity_panel.png"), fig)
end

function _azimuthal_spectrum(omega_hat::AbstractArray{<:Complex, 3})
    nfreq, NDOF = size(omega_hat, 1), size(omega_hat, 2)
    T = real(eltype(omega_hat))
    ky_1d = T.(rfftfreq(NDOF, NDOF))
    kx_1d = T.(fftfreq(NDOF, NDOF))
    k_mag = [sqrt(ky_1d[iy]^2 + kx_1d[ix]^2) for iy in 1:nfreq, ix in 1:NDOF]
    k_max = floor(Int, maximum(k_mag))
    power  = mean(abs2.(omega_hat); dims=3)[:, :, 1]
    spec   = zeros(T, k_max)
    counts = zeros(Int, k_max)
    for iy in 1:nfreq, ix in 1:NDOF
        k = round(Int, k_mag[iy, ix])
        1 <= k <= k_max || continue
        spec[k]   += power[iy, ix]
        counts[k] += 1
    end
    k_vals   = findall(>(0), counts)
    spec_avg = [spec[k] / counts[k] for k in k_vals]
    return k_vals, spec_avg
end

function plot_energy_spectrum(gen_omega_hat::AbstractArray{<:Complex, 3},
                              gt_omega_hat::AbstractArray{<:Complex, 3},
                              model_dir::AbstractString)
    mkpath(model_dir)
    k_gen, s_gen = _azimuthal_spectrum(gen_omega_hat)
    k_gt,  s_gt  = _azimuthal_spectrum(gt_omega_hat)
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel="Wavenumber k", ylabel="|ω̂(k)|²",
               title="Energy Spectrum", xscale=log10, yscale=log10)
    lines!(ax, k_gen, s_gen; label="Model")
    lines!(ax, k_gt,  s_gt;  label="Ground truth")
    axislegend(ax)
    save(joinpath(model_dir, "energy_spectrum.png"), fig)
end

end
