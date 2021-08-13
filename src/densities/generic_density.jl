# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    GenericDensity{F<:Function} <: AbstractDensity

**Deprecated**
"""
struct GenericDensity{F<:Function} <: AbstractDensity
    f::F
end

Base.convert(::Type{GenericDensity}, f::Function) = GenericDensity(f)
Base.convert(::Type{AbstractDensity}, f::Function) = GenericDensity(f)

Base.parent(density::GenericDensity) = density.f

function DensityInterface.logdensityof(density::GenericDensity, v::Any)
    logvalof(density.f(v))
end



Base.@deprecated Base.convert(::Type{AbstractDensity}, nt::NamedTuple{(:logdensity,)}) logfuncdensity(nt.logdensity)



"""
    BAT.LogFuncDensityWithGrad{F<:Function} <: AbstractDensity

*BAT-internal, not part of stable public API.*

Constructors:

    LogFuncDensityWithGrad(logf::Function, valgradlogf::Function)

A density defined by a function that computes it's logarithmic value at given
points, as well as a function that computes both the value and the gradient.

It must be safe to execute both functions in parallel on multiple threads and
processes.
"""
struct LogFuncDensityWithGrad{F<:Function,G<:Function} <: AbstractDensity
    logf::F
    valgradlogf::G
end

DensityInterface.logdensityof(density::LogFuncDensityWithGrad) = density.logf

function DensityInterface.logdensityof(density::LogFuncDensityWithGrad, v::Any)
    density.logf(v)
end

function ChainRulesCore.rrule(::typeof(DensityInterface.logdensityof), density::LogFuncDensityWithGrad, v)
    value, gradient = density.valgradlogf(v)
    @assert value isa Real
    function lfdwg_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        @assert ΔΩ isa Real
        tangent = gradient * ΔΩ
        (NoTangent(), ZeroTangent(), tangent)
    end
    return value, lfdwg_pullback
end

vjp_algorithm(density::LogFuncDensityWithGrad) = ZygoteAD()


function Base.show(io::IO, density::LogFuncDensityWithGrad)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, density.logf)
    print(io, ")")
end
