# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, StableRNGs
using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, PDMats, StatsBase

struct _TestDensityStruct <: AbstractDensity
end

struct _UniformDensityStruct <: AbstractDensity
end

struct _DeltaDensityStruct <: AbstractDensity
end

struct _ShapeDensityStruct <: AbstractDensity
end

@testset "abstract_density" begin
    mvn = MvNormal(ones(3), PDMat(Matrix{Float64}(I,3,3)))
    ValueShapes.totalndof(td::_TestDensityStruct) = Int(3)
    BAT.sampler(td::_TestDensityStruct) = BAT.sampler(mvn)
    DensityInterface.logdensityof(density::_TestDensityStruct, v::Any) = Distributions.logpdf(mvn, v)

    td = _TestDensityStruct()
    tds = BAT.DensityWithShape(td, ArrayShape{Real}(3))

    @test @inferred(isequal(@inferred(BAT.var_bounds(td)), missing))
    @test @inferred(isequal(@inferred(varshape(td)), missing))

    x = rand(3)
    @test_throws ArgumentError logvalof(td, [Inf, Inf, Inf])
    @test_throws BAT.DensityEvalException logvalof(tds, [Inf, Inf, Inf])
    @test_throws BAT.DensityEvalException logvalof(tds, [NaN, NaN, NaN])
    @test_throws ArgumentError logvalof(tds, rand(length(mvn)+1))
    @test_throws ArgumentError logvalof(tds, rand(length(mvn)-1))

    @test @inferred(logvalof(tds, x)) == @inferred(logpdf(mvn, x))


    mvu = product_distribution([Uniform() for i in 1:3])
    BAT.varshape(ud::_UniformDensityStruct) = varshape(mvu)
    DensityInterface.logdensityof(ud::_UniformDensityStruct, v::Any) = logpdf(mvu, v)
    BAT.var_bounds(ud::_UniformDensityStruct) = BAT.HyperRectBounds(BAT.HyperRectVolume(zeros(3), ones(3)))
    ValueShapes.totalndof(ud::_UniformDensityStruct) = Int(3)

    x = [-Inf, 0, Inf]
    ud_shape_1 = NamedTupleShape(a=ArrayShape{Real}(1), b=ArrayShape{Real}(1), c=ArrayShape{Real}(1))
    ud_shape_2 = NamedTupleShape(a=ArrayShape{Real}(3))
    ud = _UniformDensityStruct()

    @test @inferred(logvalof(ud, x)) == -Inf
    @test @inferred(logvalof(ud_shape_1(ud), ud_shape_1(x))) == @inferred(logpdf(mvu, x))
    @test @inferred(logvalof(ud_shape_2(ud), ud_shape_2(x))) == @inferred(logpdf(mvu, x))

    @test_throws ArgumentError logvalof(ud, vcat(x,x))

    @test logvalof(ud, x .- eps(1.0)) == -Inf

    @test_throws ArgumentError @inferred(logvalof(ud, [0 0 0]))

    ntshape = NamedTupleShape(a=ScalarShape{Real}(), b=ScalarShape{Real}(), c=ScalarShape{Real}())
    shapedasnt = ShapedAsNT(x, ntshape)

    cvd = ConstValueDist(0)
    ValueShapes.totalndof(dd::_DeltaDensityStruct) = Int(1)
    DensityInterface.logdensityof(dd::_DeltaDensityStruct, v::Any) = Distributions.logpdf(cvd, v)

    dd = _DeltaDensityStruct()
    dds = BAT.DensityWithShape(dd, ScalarShape{Real}())
    @test_throws ArgumentError logvalof(dd, 0)
    @test_throws BAT.DensityEvalException logvalof(dds, 0)

    ntdist = NamedTupleDist(a=mvn, b=mvu)
    ValueShapes.varshape(sd::_ShapeDensityStruct) = varshape(ntdist)
    DensityInterface.logdensityof(sd::_ShapeDensityStruct, v) = logpdf(ntdist, v)

    x1_for_sd = rand(length(ntdist.a))
    x2_for_sd = rand(length(ntdist.b))
    x_for_sd = vcat(x1_for_sd, x2_for_sd)

    correct_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)), b=ArrayShape{Real}(length(ntdist.b)))
    incorrect_shape_of_sd = NamedTupleShape(a=ArrayShape{Real}(length(ntdist.a)-1), b=ArrayShape{Real}(length(ntdist.b)+1))

    x_for_sd_good_shape = correct_shape_of_sd(x_for_sd)
    x_for_sd_bad_shape = incorrect_shape_of_sd(x_for_sd)

    sd = _ShapeDensityStruct()

    @test logvalof(sd, x_for_sd_good_shape) == logpdf(mvn, x1_for_sd) + logpdf(mvu, x2_for_sd)
    @test_throws ArgumentError logvalof(sd, x_for_sd_bad_shape)

    @testset "rand" begin
        td = _TestDensityStruct()
        @test rand(StableRNG(7002), BAT.bat_sampler(td)) ≈ [2.386799038, 1.072161895, 0.791486531]
    end
end
