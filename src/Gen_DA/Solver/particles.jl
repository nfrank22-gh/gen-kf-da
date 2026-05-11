function bilinear_interp_periodic(field, xp, yp)
    N  = Int32(size(field, 1))
    dx = Float32(L / N)

    fx = mod.(xp, Float32(N) * dx) ./ dx
    fy = mod.(yp, Float32(N) * dx) ./ dx

    # floor.(Int32, x) uses Reactant's traced floor(::Type{T}, ::TracedRNumber) overload
    i0 = floor.(Int32, fx) .+ Int32(1)
    j0 = floor.(Int32, fy) .+ Int32(1)
    wx = fx .- floor.(fx)
    wy = fy .- floor.(fy)
    i1 = mod.(i0, N) .+ Int32(1)
    j1 = mod.(j0, N) .+ Int32(1)

    # Column-major linear indices: (col-1)*nrows + row
    linidx(j, i) = @. (i - Int32(1)) * N + j
    v00 = field[linidx(j0, i0)]
    v10 = field[linidx(j0, i1)]
    v01 = field[linidx(j1, i0)]
    v11 = field[linidx(j1, i1)]

    return @. (1f0-wx)*(1f0-wy)*v00 + wx*(1f0-wy)*v10 +
              (1f0-wx)*wy*v01 + wx*wy*v11
end

function tracer_substep(xp, yp, h_xp, h_yp, u, v, beta_i, gamma_i, dt)
    h_xp = @. u + beta_i * h_xp
    h_yp = @. v + beta_i * h_yp
    xp   = @. xp + gamma_i * dt * h_xp
    yp   = @. yp + gamma_i * dt * h_yp
    return xp, yp, h_xp, h_yp
end
