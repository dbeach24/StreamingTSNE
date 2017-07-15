

module OnlineTSNE

using ProgressMeter
using HDF5
using StaticArrays
using VPTrees
#using RegionTrees

import Base: push!

const Sample = Int

# type TSNEState{P}
#     niters::Int
#     samples::Vector{Sample}
#     embedding::Vector{SVector{P}}
#     numiters::Vector{Int}
#     dist::Function
#     #vptree::VPTree

# end

# function tsne_state(metric::Function, P::Int=2, niters::Int=1000)
#     samples = Vector{Sample}()
#     TSNEState{T,P,typeof(dist)}(
#         niters,
#         samples,
#         Vector{SVector{P}}(),
#         Vector{Int}(),
#         dist,
#         #VPTree(metric, samples)
#     )
# end


# function push!(tsne::TSNEState, sample::UInt64)
#     push!(tsne.samples, sample)
#     push!(tsne.embedding, init_embedding(tsne))
#     push!(tsne.numiters, 0)
# end



function solve_beta(vptree::VPTree, i::Sample, perplexity::Number=30.0;
    β0::Number=-1, max_iter::Int=50, tol::Number=0.03)

    # NOTE: This loop solves for sigma (σ) in terms of beta (β)
    # where β = 1 / 2σ^2

    if β0 < 0
        (_, σ0) = closest(vptree, i)
        β0 = 1 / (2σ0^2)
    end

    β = β0
    βmin = 0.0
    βmax = Inf

    logU = log(perplexity)

    # exclude i from distances
    # update all other distances by subtracting minD
    # for each distance D[i,j], P[j|i] = exp(-β * D[i,j])
    # ΣP = sum(P)
    # H = -β + log(sumP) + β * sum(D[j] * P[j]) / sumP

    for tries in 1:max_iter

        ΣP = 0.0
        DdotP = 0.0
        σ = sqrt(1 / (2β))
        τ = 3σ
        search(vptree, i, τ) do j, d
            if i != j
                d2 = d^2
                Pj = exp(-β * d2)
                ΣP += Pj
                DdotP += d2 * Pj
            end
            # continue search with radius=τ
            τ
        end
        H = log(ΣP) + β * DdotP / ΣP
        Hdiff = H - logU
        abs(Hdiff) < tol && break

        if Hdiff > 0.0
            βmin = β
            β = isfinite(βmax) ? (β + βmax) / 2 : 2β
        else
            βmax = β
            β = (β + βmin) / 2
        end

    end

    return β
end


function cached_distance(dist::Function, maxsize::Int=10000)

    cache = Dict{Tuple{Int,Int}, Float64}()

    function cachedist(i, j)
        i == j && return 0.0
        i > j && ((i, j) = (j, i))
        key = (i,j)
        d = get(cache, key, -1.0)
        d >= 0.0 && return d
        d = dist(i,j)
        length(cache) > maxsize && empty!(cache)
        cache[key] = d
        d
    end

end


function build_tree(X)
    D, N = size(X)

    dist = cached_distance(10000) do i,j
        norm(view(X,:,i) - view(X,:,j))
    end

    make_vp_tree(dist, collect(1:N))
end

function test(X)
    D, N = size(X)

    println("building vp tree...")
    tree = build_tree(X)

    println("solving for neighborhoods...")
    for i = 1:N
        β = solve_beta(tree, i)
        #println(β)
    end

end


function main(infile)
    X = map(Float64, h5read(infile, "X"))
    test(X)
end

main("samples/mnist_4000.h5")

end # module




