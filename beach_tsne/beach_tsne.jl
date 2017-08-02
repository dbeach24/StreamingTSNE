

#module OnlineTSNE

using ProgressMeter
using Distributions
using HDF5
using StaticArrays
using VPTrees
using QuadTrees
using SFML

import Base: insert!, count
import QuadTrees: QTNode, QTNodePtr

const Vec = Vector{Float64}
const Point = SVector{2, Float64}
const Sample = Int

const InitDist = MultivariateNormal(zeros(2), eye(2) * 1e-4)

immutable TSNENeighbor
    dist::Float64
    pij::Float64
end

type TSNEPoint
    id::Int
    x::Vec
    y::Point
    Δy::Point
    ∂y::Point
    gain::Point
    iterctr::Int
    freshness::Float64 # ???
    β::Float64
    ΣP::Float64
    neighbors::Dict{Int, TSNENeighbor}
end


type TSNEState
    niters::Int
    perplexity::Float64
    θ::Float64
    dist::Function
    points::Dict{Int, TSNEPoint}
    vptree::VPTree
    quadtree::QuadTree
    nextid::Int
    ΣPij::Float64
end

count(state::TSNEState) = count(state.vptree)

function cached_distance(dist::Function, maxsize::Int=1000000)

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
        id, value, pt, SVector(0.0, 0.0), SVector(0.0, 0.0), SVector(1.0, 1.0), 0, 0.0, -1.0, -1.0,
        Dict{Int, TSNENeighbor}()
    )
end


function make_tsne_state(; niters=1000, perplexity=20.0, θ=0.5)

    points = Dict{Int, TSNEPoint}()

    dist = cached_distance() do i,j
        norm(points[i].x .- points[j].x)
    end

    vptree = make_vp_tree(dist)
    quadtree = make_quad_tree(Point(-1000.0, -1000.0), Point(1000.0, 1000.0))
    nextid = 1
    ΣPij = 0.0
    state = TSNEState(
        niters,
        perplexity,
        θ,
        dist,
        points,
        vptree,
        quadtree,
        nextid,
        ΣPij
    )

end


function insert!(state::TSNEState, value::Vec)
    id = state.nextid
    state.nextid += 1

    state.points[id] = pi = make_tsne_point(id, value)
    insert!(state.vptree, id)
    insert!(state.quadtree, id, pi.y)
end


function update_neighborhood!(state::TSNEState, i::Sample)
    pi = state.points[i]
    β, ΣP, neighbor_dist = solve_neighborhood(state, i; β0 = pi.β)
    pi.β = β
    pi.ΣP = ΣP

    updated = Set{Int}()

    ΔΣPij = 0.0

    for (j, d) ∈ neighbor_dist
        pj = state.points[j]
        ΔΣPij += update_neighbor!(pi, pj, d)
        push!(updated, j)
    end

    for (j, neighbor) ∈ pi.neighbors
        if j ∉ updated
            pj = state.points[j]
            d = neighbor.dist
            ΔΣPij += update_neighbor!(pi, pj, d)
        end
    end

    state.ΣPij += ΔΣPij
    nothing
end


function solve_neighborhood(state::TSNEState, i::Sample;
                            β0::Number=-1, max_iter::Int=50, tol::Number=1e-5)

    # NOTE: This loop solves for sigma (σ) in terms of beta (β)
    # where β = 1 / 2σ^2
    # σ = sqrt(1 / 2β)

    u = state.perplexity

    neighbor_dist = closest(state.vptree, i, floor(3u))

    if β0 < 0
        (_, σ0) = neighbor_dist[1]
        β0 = 1 / (2σ0^2)
    end

    β = β0
    βmin = 0.0
    βmax = Inf
    ΣP = 0
    lastn = 1

    logU = log(u)
    logUtol = logU * tol

    # for each distance D[i,j], P[j|i] = exp(-β * D[i,j])
    # ΣP = sum(P)
    # H = -β + log(sumP) + β * sum(D[j] * P[j]) / sumP

    pi = state.points[i]
    for iter ∈ 1:max_iter

        ΣP = 0.0
        DdotP = 0.0
        σ = sqrt(1 / 2β)
        τ = 3σ

        for (n, (j, d)) in enumerate(neighbor_dist)
            if d > τ
                lastn = n - 1
                break
            end
            d2 = d^2
            Pj = exp(-β * d2)
            ΣP += Pj
            DdotP += d2 * Pj
        end

        # H is the perplexity of the gaussian on this iteration
        H = log(ΣP) + β * DdotP / ΣP
        Hdiff = H - logU

        #println("iter $iter, Hdiff=$Hdiff, β = $β, σ=$σ")

        abs(Hdiff) < logUtol && break

        if Hdiff > 0.0
            βmin = β
            β = isfinite(βmax) ? (β + βmax) / 2 : 2β
        else
            βmax = β
            β = (β + βmin) / 2
        end

    end

    return β, ΣP, neighbor_dist[1:lastn]
end

function update_neighbor!(pi::TSNEPoint, pj::TSNEPoint, dist::Float64)
    d2 = dist^2
    pj_i = exp(-pi.β * d2) / pi.ΣP
    pi_j = exp(-pj.β * d2) / pj.ΣP
    pij = (pj_i + pi_j) / 2
    if pij < 1e-3
        pij = 0.0
    end
    old_pij = get(pi.neighbors, pj.id, TSNENeighbor(0.0, 0.0)).pij
    pi.neighbors[pj.id] = TSNENeighbor(dist, pij)
    pj.neighbors[pi.id] = TSNENeighbor(dist, pij)
    delta_pij = 2(pij - old_pij)
    delta_pij
end

pj_given_i(state::TSNEState, pj::TSNEPoint, pi::TSNEPoint) = exp(-pi.β * state.dist(pi.id, pj.id)^2) / pi.ΣP
pij(state::TSNEState, pi::TSNEPoint, pj::TSNEPoint) = ((
        pj_given_i(state, pj, pi) + pj_given_i(state, pi, pj)
    ) / 2*count(state)) / count(state)^2

qijZ(pi::TSNEPoint, pj::TSNEPoint) = qijZ(pi.y, pj.y)
qijZ(yi::Point, yj::Point) = 1 / (1 + xydist2(yi, yj))

xydist2(yi::Point, yj::Point) = (yi[:1] - yj[:1])^2 + (yi[:2] - yj[:2])^2
xydist2(pi::TSNEPoint, pj::TSNEPoint) = xydist2(pi.y, pj.y)

"""Attractive Force (Barnes-Hut)"""
function f_attr(state::TSNEState, i::Sample)
    pi = state.points[i]
    f = SVector(0.0, 0.0)
    neighbors = pi.neighbors
    for (j, neighbor) in neighbors
        pj = state.points[j]
        pij = neighbor.pij
        fmag = pij * qijZ(pi, pj)
        @assert fmag ≥ 0
        f += (pi.y - pj.y) * fmag
    end
    f / state.ΣPij
end

"""Z-term (Barnes-Hut)"""
function Zterm(state::TSNEState)
    root = state.quadtree.root
    sum(Zterm(state, pi, root) for pi in values(state.points))
end

Zterm(state::TSNEState, pi::TSNEPoint, n::QTNodePtr) = isnull(n) ? 0.0 : Zterm(state, pi, get(n))

function Zterm(state::TSNEState, pi::TSNEPoint, node::QTNode)
    c = count(node)
    yi = pi.y
    yj = value(node)

    if node.id == pi.id
        0.0
    elseif count(node) == 1 || bh_eligible(state, pi, node)
        c * qijZ(yi, yj)
    else
        (
            Zterm(state, pi, node.ul) +
            Zterm(state, pi, node.ur) +
            Zterm(state, pi, node.ll) +
            Zterm(state, pi, node.lr)
        )
    end
end

"""Repulsive Force (Barnes-Hut)"""
function f_rep(state::TSNEState, i::Sample, Z::Float64) 
    f = f_rep(state, state.points[i], state.quadtree.root)
    f / Z
end

f_rep(state::TSNEState, pi::TSNEPoint, n::QTNodePtr) = isnull(n) ? SVector(0.0, 0.0) : f_rep(state, pi, get(n))

function f_rep(state::TSNEState, pi::TSNEPoint, node::QTNode)
    c = count(node)
    yi = pi.y
    yj = value(node)

    if node.id == pi.id
        SVector(0.0, 0.0)
    elseif c == 1 || bh_eligible(state, pi, node)
        # use center of mass
        qZ = qijZ(yi, yj)
        q2Z2 = qZ^2
        @assert q2Z2 ≥ 0
        c * q2Z2 * (yi - yj)
    else
        (
            f_rep(state, pi, node.ul) +
            f_rep(state, pi, node.ur) +
            f_rep(state, pi, node.ll) +
            f_rep(state, pi, node.lr)
        )
    end
end

function bh_eligible(state::TSNEState, point::TSNEPoint, node::QTNode)
    yi = point.y
    yj = value(node)
    r_cell = sqrt(xydist2(node.p1, node.p2))
    #y_cell = xydist2(yi, yj)
    # TODO SEEMS WRONG
    # Think it should be:
    y_cell = sqrt(xydist2(yi, yj))

    (r_cell / y_cell) < state.θ
end


objective(state::TSNEState) = objective(state, Zterm(state))

function objective(state::TSNEState, Z)
    ΣPij = state.ΣPij
    kl = 0.0
    for (i, pᵢ) ∈ state.points
        for (j, neighbor) ∈ pᵢ.neighbors
            pⱼ = state.points[j]
            pᵢⱼ = neighbor.pij / ΣPij
            if pᵢⱼ > 0.0
                qᵢⱼ = qijZ(pᵢ, pⱼ) / Z
                kl += pᵢⱼ * log(pᵢⱼ / qᵢⱼ)
            end
        end
    end
    kl
end

"""Gradient (Barnes-Hut)"""
function gradient(state::TSNEState, i::Sample, Z, exaggeration)
    fₐ = f_attr(state, i) * exaggeration
    fᵣ = f_rep(state, i, Z)
    4 * (fₐ - fᵣ)
end

update!(state::TSNEState) = update!(state, keys(state.points))

function update!(state::TSNEState, indices)

    Z = Zterm(state)
    #println("Z = $Z")

    # compute the gradients
    for i in indices
        pi = state.points[i]
        exaggeration = max(1.0, 4.0 - pi.iterctr * .01)
        #exaggeration = ifelse(pi.iterctr < 100, 4.0, 1.0)
        pi.∂y = gradient(state, i, Z, exaggeration)
    end

    # if more than 1/3 of points are updated,
    # just rebuild the quadtree
    if 3*length(indices) > count(state)
        newtree = make_quad_tree(Point(-1000.0, -1000.0), Point(1000.0, 1000.0))
        remove_points = false
    else
        # otherwise, update quadtree in-place
        newtree = state.QuadTree
        remove_points = true
    end

    η = 500.0 # learning rate

    # update point states
    for i in indices
        pi = state.points[i]
        α = ifelse(pi.iterctr < 20, 0.5, 0.8)

        # update gain
        g1 = update_gain(pi.gain[:1], pi.∂y[:1], pi.Δy[:1])
        g2 = update_gain(pi.gain[:2], pi.∂y[:2], pi.Δy[:2])
        pi.gain = Point(g1, g2)

        # compute updated velocity
        pi.Δy = α * pi.Δy - η * (pi.gain .* pi.∂y)

        # update point's state
        remove_points && delete!(state.quadtree, i, pi.y)
        pi.y += pi.Δy
        pi.iterctr += 1
        insert!(newtree, i, pi.y)

    end

    state.quadtree = newtree

    nothing
end

update_gain(gain, dy, iy) = max(0.01, sign(dy) != sign(iy) ? gain + 0.2 : gain * 0.8)

function load_tsne(X)

    D, N = size(X)

    state = make_tsne_state(perplexity=20., θ=0.8)

    for i ∈ 1:N
        vec = X[:,i]
        insert!(state, X[:,i])
    end

    for i ∈ 1:N
        println("updating β for point $i")
        update_neighborhood!(state, i)
    end

    #psum = sum(pij(state, pi, pj) for pi in values(state.points), pj in values(state.points) if pi.id ≠ pj.id)
    #@show ps = psum(state)
    #@assert abs(psum - 1.0) < 0.1

    state
end    


function mainloop(state, labels, iters=1000)
    window = mkwindow()

    for t ∈ 1:iters
        println("update #$t")
        update!(state)
        view(window, state, labels) || break
    end
end

const window_width = 1500
const window_height = 1500
const hwidth = window_width / 2
const hheight = window_height / 2
const colors = [
    SFML.Color(141,211,199),
    SFML.Color(255,255,179),
    SFML.Color(190,186,218),
    SFML.Color(251,128,114),
    SFML.Color(128,177,211),
    SFML.Color(253,180,98),
    SFML.Color(179,222,105),
    SFML.Color(252,205,229),
    SFML.Color(217,217,217),
    SFML.Color(188,128,189),
]
const viewscale = 100

function mkdot(x, y; r=8, label=1)
    dot = CircleShape()
    set_position(dot, Vector2f(x, y))
    set_radius(dot, r)
    set_fillcolor(dot, colors[label])
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
    sx = (x * hwidth / viewscale) + hwidth
    sy = (y * hheight / viewscale) + hheight
    Point(sx, sy)
end

function view(window, state, labels)

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
        dot = mkdot(xy[:1], xy[:2]; r=8, label=labels[point.id]+1)
        draw(window, dot)
    end

    display(window)
    true

end

function main(fname)
    X = map(Float64, h5read(fname, "X"))
    labels = h5read(fname, "labels")

    # normalize pixel values to 0..1
    X ./= 255.0

    # Z-score normalization seems like a bad idea for this data!
    # for d in 1:size(X,1)
    #     sd = std(X[d,:])
    #     if sd > 0
    #         X[d,:] ./= sd
    #     end
    # end
    state = load_tsne(X)
    mainloop(state, labels)
    state
end

state = main("../samples/mnist_2000.h5")

#end # module




