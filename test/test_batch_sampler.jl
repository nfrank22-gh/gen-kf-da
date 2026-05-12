using Test
using Gen_DA.NN
using Random

@testset "BatchSampler: output shapes" begin
    rng = MersenneTwister(1)
    s = BatchSampler(rng, 8, 4, 3, 10, 16; fix_x=false, fix_thetas=false)
    x_all, thetas_all = sample_epoch!(s)
    @test size(x_all)      == (8, 4 * 3)
    @test size(thetas_all) == (10 * 3, 16)
end

@testset "BatchSampler: get_batch indexing" begin
    rng = MersenneTwister(2)
    s = BatchSampler(rng, 8, 4, 3, 10, 16; fix_x=false, fix_thetas=false)
    x_all, thetas_all = sample_epoch!(s)
    for i in 1:3
        x, thetas, cols = get_batch(s, x_all, thetas_all, i)
        @test size(x)      == (8, 4)
        @test size(thetas) == (10, 16)
        @test cols == (i-1)*4+1 : i*4
        @test x      == x_all[:, cols]
        @test thetas == thetas_all[(i-1)*10+1 : i*10, :]
    end
end

@testset "BatchSampler: fix_x caches array" begin
    rng = MersenneTwister(3)
    s = BatchSampler(rng, 8, 4, 3, 10, 16; fix_x=true, fix_thetas=false)
    x1, _ = sample_epoch!(s)
    x2, _ = sample_epoch!(s)
    @test x1 === x2
end

@testset "BatchSampler: fix_thetas caches array" begin
    rng = MersenneTwister(4)
    s = BatchSampler(rng, 8, 4, 3, 10, 16; fix_x=false, fix_thetas=true)
    _, t1 = sample_epoch!(s)
    _, t2 = sample_epoch!(s)
    @test t1 === t2
end

@testset "BatchSampler: resample when not fixed" begin
    rng = MersenneTwister(5)
    s = BatchSampler(rng, 8, 4, 3, 10, 16; fix_x=false, fix_thetas=false)
    x1, t1 = sample_epoch!(s)
    x2, t2 = sample_epoch!(s)
    @test x1 !== x2
    @test t1 !== t2
end
