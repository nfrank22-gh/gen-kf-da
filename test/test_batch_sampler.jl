using Test
using Gen_DA.NN
using Random

@testset "BatchSampler: output shapes" begin
    rng = MersenneTwister(1)
    s = BatchSampler(rng, 4, 3, 10, 16; fix_thetas=false)
    sample_epoch!(s)
    @test size(s._thetas_cache) == (10 * 3, 16)
end

@testset "BatchSampler: get_batch indexing" begin
    rng = MersenneTwister(2)
    s = BatchSampler(rng, 4, 3, 10, 16; fix_thetas=false)
    sample_epoch!(s)
    for i in 1:3
        thetas, cols = get_batch(s, i)
        @test size(thetas) == (10, 16)
        @test cols == (i-1)*4+1 : i*4
        @test thetas == s._thetas_cache[(i-1)*10+1 : i*10, :]
    end
end

@testset "BatchSampler: fix_thetas caches array" begin
    rng = MersenneTwister(4)
    s = BatchSampler(rng, 4, 3, 10, 16; fix_thetas=true)
    sample_epoch!(s)
    t1 = s._thetas_cache
    sample_epoch!(s)
    t2 = s._thetas_cache
    @test t1 === t2
end

@testset "BatchSampler: resample when not fixed" begin
    rng = MersenneTwister(5)
    s = BatchSampler(rng, 4, 3, 10, 16; fix_thetas=false)
    sample_epoch!(s)
    t1 = s._thetas_cache
    sample_epoch!(s)
    t2 = s._thetas_cache
    @test t1 !== t2
end
