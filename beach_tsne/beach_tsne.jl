

module OnlineTSNE

using ProgressMeter
using Distributions
using HDF5
using StaticArrays
using VPTrees
using QuadTrees
using SFML

import Base: insert!, count

const Vec = Vector{Float64}
const Point = SVector{2, Float64}
const Sample = Int

const InitDist = MultivariateNormal(zeros(2), eye(2) * 1e-4)

type TSNEPoint
    id::Int
    x::Vec
    y::Point
    priory::Point
    iterctr::Int
    freshness::Float64 # ???
    β::Float64
    ΣP::Float64
end


type TSNEState
    niters::Int
    perplexity::Float64
    dist::Function
    points::Dict{Int, TSNEPoint}
    vptree::VPTree
    quadtree::QuadTree
    nextid::Int
end

count(state::TSNEState) = count(state.vptree)

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

function make_tsne_point(id::Sample, value::Vec)
    pt = rand(InitDist)
    TSNEPoint(
        id, value, pt, pt, 0, 0.0, -1.0, -1.0
    )
end


function make_tsne_state(; niters=1000, perplexity=20.0)

    points = Dict{Int, TSNEPoint}()

    dist = cached_distance() do i,j
        norm(points[i].x - points[j].x)
    end

    vptree = make_vp_tree(dist)
    quadtree = make_quad_tree(Point(-100.0, -100.0), Point(100.0, 100.0))
    nextid = 1
    state = TSNEState(
        niters,
        perplexity,
        dist,
        points,
        vptree,
        quadtree,
        nextid
    )

end


function insert!(state::TSNEState, value::Vec)
    id = state.nextid
    state.nextid += 1

    state.points[id] = point = make_tsne_point(id, value)
    insert!(state.vptree, id)
    insert!(state.quadtree, id, point.y)
end


function updateβ!(state::TSNEState, i::Sample)
    point = state.points[i]
    β, ΣP = solveβ(state.vptree, i, state.perplexity; β0 = point.β)
    point.β = β
    point.ΣP = ΣP
    nothing
end


function solveβ(vptree::VPTree, i::Sample, perplexity::Number=30.0;
    β0::Number=-1, max_iter::Int=50, tol::Number=0.03)

    # NOTE: This loop solves for sigma (σ) in terms of beta (β)
    # where β = 1 / 2σ^2
    # σ = sqrt(1 / 2β)

    if β0 < 0
        (_, σ0) = closest(vptree, i)
        β0 = 1 / (2σ0^2)
    end

    β = β0
    βmin = 0.0
    βmax = Inf
    ΣP = 0

    logU = log(perplexity)

    # exclude i from distances
    # update all other distances by subtracting minD
    # for each distance D[i,j], P[j|i] = exp(-β * D[i,j])
    # ΣP = sum(P)
    # H = -β + log(sumP) + β * sum(D[j] * P[j]) / sumP

    for iter ∈ 1:max_iter

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
        # H is the perplexity of the gaussian on this iteration
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

    return β, ΣP
end

pj_given_i(state::TSNEState, j::TSNEPoint, i::TSNEPoint) = exp(-i.β * state.dist(i.id, j.id)^2) / i.ΣP
pij(state::TSNEState, i::TSNEPoint, j::TSNEPoint) = (pj_given_i(state, j, i) + pj_given_i(state, i, j)) / 2*count(state)

xydist2(x::Point, y::Point) = (x[:1] - y[:1])^2 + (x[:2] - y[:2])^2
xydist2(x::TSNEPoint, y::TSNEPoint) = xydist2(x.y, y.y)

function qij(i::TSNEPoint, j::TSNEPoint)
    denom = sum(1/(1 + xydist2(k, l)) for k in all for l in all if k ≠ l)
    (1 / (1 + xydist2(i, j))) / denom
end

function f_attr(state::TSNEState, i::Sample)
    pointi = state.points[i]
    β = pointi.β
    σ = sqrt(1 / 2β)
    τ = 3σ
    fx = 0.0
    fy = 0.0
    search(state.vptree, i, τ) do j, dist
        if j ≠ i
            pointj = state.points[j]
            fmag = pij(state, pointi, pointj) / (1 + xydist2(pointi, pointj))
            fx += (pointi.y[:1] .- pointj.y[:1]) .* fmag
            fy += (pointi.y[:2] .- pointj.y[:2]) .* fmag
        end
        τ
    end
    SVector(fx, fy)
end

function f_rep(state::TSNEState, i::Sample)
    pointi = state.points[i]
    fx = 0.0
    fy = 0.0


    SVector(fx, fy)
end

gradient(state::TSNEState, i::Sample) = 4 * (f_attr(state, i) - f_rep(state, i))

update!(state::TSNEState) = update!(state, keys(state.points))

function update!(state::TSNEState, indices)

    # compute the gradients
    grad = Dict{Int, Point}()
    for i in indices
        grad[i] = gradient(state, i)
    end

    # update point states
    for i in indices
        point = state.points[i]
        η = -1e-3
        α = ifelse(point.iterctr < 250, 0.5, 0.8)

        ∂y = grad[i]

        # compute updated points
        y_tm2 = point.priory
        y_tm1 = point.y
        y_t = y_tm1 .+ η .* ∂y .+ α .* (y_tm1 .- y_tm2)

        # update point's state
        point.y = y_t
        point.priory = y_tm1
        point.iterctr += 1
    end
    nothing
end

function main(infile)
    X = map(Float64, h5read(infile, "X"))

    D, N = size(X)

    state = make_tsne_state()

    for i ∈ 1:N
        vec = X[:,i]
        insert!(state, X[:,i])
    end

    for i ∈ 1:N
        println("updating β for point $i")
        updateβ!(state, i)
    end

    window = mkwindow()

    for t ∈ 1:1000
        println("update #$t")
        update!(state)
        # for i ∈ 1:N
        #     update!(state, i)
        # end

        view(window, state) || break

    end

end

const window_width = 1000
const window_height = 1000
const hwidth = window_width / 2
const hheight = window_height / 2

function mkdot(x, y; r=2, color=SFML.red)
    dot = CircleShape()
    set_position(dot, Vector2f(x, y))
    set_radius(dot, r)
    set_fillcolor(dot, color)
    set_origin(dot, Vector2f(r, r))
    dot
end

function mkwindow()
    window = RenderWindow("t-SNE", window_width, window_height)
    set_framerate_limit(window, 60)
    set_vsync_enabled(window, true)
    window
end

function vec2screen(vec::Point)
    x, y = vec
    sx = (x * hwidth) + hwidth
    sy = (y * hheight) + hheight
    Point(sx, sy)
end

function view(window, state)

    event = Event()

    while pollevent(window, event)
        if get_type(event) == EventType.CLOSED
            close(window)
            return false
        end
    end

    clear(window, SFML.white)

    for point in values(state.points)
        xy = vec2screen(point.y)
        dot = mkdot(xy[:1], xy[:2])
        draw(window, dot)
    end

    display(window)
    true

end

main("../samples/mnist_250.h5")

end # module




