using Zygote
using Base: copymutable, Perm, ord, Ordering
using Base.Sort: Algorithm, DEFAULT_UNSTABLE
import Base: invperm, sortperm, Forward


function sortperm(A::AbstractArray;
    alg::Algorithm=DEFAULT_UNSTABLE,
    lt=isless,
    by=identity,
    rev::Union{Bool,Nothing}=nothing,
    order::Ordering=Forward,
    dims...) #to optionally specify dims argument
ordr = ord(lt,by,rev,order)
if ordr === Forward && isa(A,Vector) && eltype(A)<:Integer
n = length(A)
if n > 1
min, max = extrema(A)
(diff, o1) = sub_with_overflow(max, min)
(rangelen, o2) = add_with_overflow(diff, oneunit(diff))
if !o1 && !o2 && rangelen < div(n,2)
  return sortperm_int_range(A, rangelen, min)
end
end
end
ix = copymutable(LinearIndices(A))
sort!(ix; alg, order = Perm(ordr, vec(A)), dims...)
end


function invperm(a::AbstractArray)
    b = zero(a) # similar vector of zeros
    n = length(a)
    @inbounds for (i, j) in enumerate(a)
        ((1 <= j <= n) && b[j] == 0) ||
            throw(ArgumentError("argument is not a permutation"))
        b[j] = i
    end
    b
end


Zygote.@adjoint function sort(x::AbstractArray; dims=1)
    p = sortperm(x, dims=dims)
    x[p], x̄ -> (x̄[invperm(p)],)
end