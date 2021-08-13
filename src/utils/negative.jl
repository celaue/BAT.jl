# This file is a part of BAT.jl, licensed under the MIT License (MIT).



struct Negative{F<:Callable} <: Function
    orig_f::F
end

@inline (f::Negative)(x) = - f.orig_f(x)


ValueShapes.varshape(f::Negative) = varshape(f.orig_f)
ValueShapes.unshaped(f::Negative) = Negative(unshaped(f.orig_f))


function Base.show(io::IO, f::Union{LogDensityOf,Negative})
    print(io, Base.typename(typeof(f)).name, "(")
    show(io, f.orig_f)
    print(io, ")")
end

Base.show(io::IO, M::MIME"text/plain", f::Union{LogDensityOf,Negative}) = show(io, f)


function negative end
negative(f::Callable) = Negative(f)
negative(f::Negative) = f.orig_f
