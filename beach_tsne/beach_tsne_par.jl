#!/usr/bin/env julia5

#module OnlineTSNE

using ArgParse
using ProgressMeter
using Distributions
using HDF5
using StaticArrays
using VPTrees
using QuadTrees
using SFML

import Base: insert!, count
import QuadTrees: QTNode, QTNodePtr
import VPTrees: remove!

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
    zterm::Float64
    gain::Point
    iterctr::Int
    staleness::Float64
    β::Float64
    ΣP::Float64
    neighbors::Dict{Int, TSNENeighbor}
end


type TSNEState
    niters::Int
    perplexity::Float64
    θ::Float64
    dist::Function
    data::Dict{Int, Vec}
    points::Dict{Int, TSNEPoint}
    vptree::VPTree
    quadtree::QuadTree
    nextid::Int
    ΣPij::Float64
    Z::Float64
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
        id,                         # id
        value,                      # x
        pt,                         # y
        SVector(0.0, 0.0),          # Δy
        0.0,                        # zterm
        SVector(1.0, 1.0),          # gain
        0,                          # iterctr
        0.0,                        # staleness
        -1.0,                       # β
        -1.0,                       # ΣP
        Dict{Int, TSNENeighbor}()   # neighbors
    )
end


function make_tsne_state(; niters=1000, perplexity=20.0, θ=0.5)

    data = Dict{Int, Vec}()
    points = Dict{Int, TSNEPoint}()

    dist = cached_distance() do i,j
        norm(data[i] .- data[j])
    end

    #dist(i,j) = norm(points[i].x .- points[j].x)

    vptree = make_vp_tree(dist)
    quadtree = make_quad_tree(Point(-1000.0, -1000.0), Point(1000.0, 1000.0))
    nextid = 1
    ΣPij = 0.0
    Z = -1.0
    state = TSNEState(
        niters,
        perplexity,
        θ,
        dist,
        data,
        points,
        vptree,
        quadtree,
        nextid,
        ΣPij,
        Z,
    )

end


function insert!(state::TSNEState, value::Vec)
    id = state.nextid
    state.nextid += 1

    state.data[id] = value
    state.points[id] = pi = make_tsne_point(id, value)
    insert!(state.vptree, id)
    insert!(state.quadtree, id, pi.y)
    id
end

function remove!(state::TSNEState, i::Sample; neighbor_penalty::Float64=1.0)

    i ∈ keys(state.points) || error("no such point in tree!")

    points = state.points
    pi = points[i]

    ΔΣPij = 0.0

    for (j, neighbor) ∈ pi.neighbors
        pj = points[j]
        delete!(pj.neighbors, i)
        pj.staleness += neighbor_penalty
        ΔΣPij += 2neighbor.pij
    end

    state.ΣPij -= ΔΣPij

    (zterm, fr) = f_rep(state, i)
    state.Z -= zterm

    remove!(state.vptree, i)
    delete!(state.quadtree, i, pi.y)
    delete!(points, i)

    nothing
end

function update_neighborhood!(state::TSNEState, i::Sample; neighbor_penalty::Float64=1.0)
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
        pj.staleness += neighbor_penalty
    end

    for (j, neighbor) ∈ pi.neighbors
        if j ∉ updated
            pj = state.points[j]
            d = neighbor.dist
            ΔΣPij += update_neighbor!(pi, pj, d)
            pj.staleness += neighbor_penalty
        end
    end

    state.ΣPij += ΔΣPij

    pi.staleness = 0.0 # updated point is completely fresh!

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

qij(pi::TSNEPoint, pj::TSNEPoint) = qij(pi.y, pj.y)
qij(yi::Point, yj::Point) = 1 / (1 + xydist2(yi, yj))

xydist2(yi::Point, yj::Point) = (yi[:1] - yj[:1])^2 + (yi[:2] - yj[:2])^2
xydist2(pi::TSNEPoint, pj::TSNEPoint) = xydist2(pi.y, pj.y)

"""Attractive Force (Barnes-Hut)"""
function f_attr(state::TSNEState, i::Sample)
    pi = state.points[i]
    f = SVector(0.0, 0.0)
    neighbors = pi.neighbors
    for (j, neighbor) in neighbors
        pj = state.points[j]
        pᵢⱼ = neighbor.pij
        qᵢⱼ = qij(pi, pj)
        fmag = pᵢⱼ * qᵢⱼ
        @assert fmag ≥ 0
        f += (pi.y - pj.y) * fmag
    end

    # normalize attractive forces
    f /= state.ΣPij

    # add mild attraction to origin
    # promotes centering and compactness
    f += (pi.y) * 1e-7

    f
end

"""Dual-Tree Algorithm for computing Z."""
get_zterm(state::TSNEState) = get_zterm(state, state.quadtree.root, state.quadtree.root)
get_zterm(state::TSNEState, ni::QTNodePtr, nj::QTNodePtr) = (
    (isnull(ni) || isnull(nj)) ? 0.0 : get_zterm(state, get(ni), get(nj))
)
get_zterm(state::TSNEState, ni::QTNodePtr, nj::QTNode) = isnull(ni) ? 0.0 : get_zterm(state, get(ni), nj)
get_zterm(state::TSNEState, ni::QTNode, nj::QTNodePtr) = isnull(nj) ? 0.0 : get_zterm(state, ni, get(nj))

function get_zterm(state::TSNEState, ni::QTNode, nj::QTNode)
    cni = count(ni)
    cnj = count(nj)

    if cni == 0 || cnj == 0
        zterm = 0.0
    elseif cni == cnj == 1
        if ni.id != nj.id
            zterm = qij(value(ni), value(nj))
        else
            zterm = 0.0
        end
    elseif bh_eligible(state, ni, nj)
        zterm = cni * cnj * qij(value(ni), value(nj))
    else
        dni = xydist2(ni.p1, ni.p2)
        dnj = xydist2(nj.p1, nj.p2)
        if dni > dnj && cni > 1
            zterm = (
                get_zterm(state, ni.ul, nj) +
                get_zterm(state, ni.ur, nj) +
                get_zterm(state, ni.ll, nj) +
                get_zterm(state, ni.lr, nj)
            )
        else
            zterm = (
                get_zterm(state, ni, nj.ul) +
                get_zterm(state, ni, nj.ur) +
                get_zterm(state, ni, nj.ll) +
                get_zterm(state, ni, nj.lr)
            )
        end
    end
    zterm
end

"""Repulsive Force (Barnes-Hut)"""
f_rep(state::TSNEState, i::Sample) = f_rep(state, state.points[i], state.quadtree.root)
f_rep(state::TSNEState, pi::TSNEPoint, n::QTNodePtr) = isnull(n) ? (0.0, SVector(0.0, 0.0)) : f_rep(state, pi, get(n))

function f_rep(state::TSNEState, pi::TSNEPoint, node::QTNode)
    c = count(node)
    yi = pi.y
    yj = value(node)

    if node.id == pi.id
        zterm = 0.0
        f = SVector(0.0, 0.0)
    elseif c == 1 || bh_eligible(state, pi, node)
        # use center of mass
        qᵢⱼ = qij(yi, yj)
        zterm = c * qᵢⱼ
        f = zterm * qᵢⱼ * (yi - yj)
    else
        z1, f1 = f_rep(state, pi, node.ul)
        z2, f2 = f_rep(state, pi, node.ur)
        z3, f3 = f_rep(state, pi, node.ll)
        z4, f4 = f_rep(state, pi, node.lr)
        zterm = z1 + z2 + z3 + z4
        f = f1 + f2 + f3 + f4
    end
    (zterm, f)
end

function bh_eligible(state::TSNEState, pi::TSNEPoint, node::QTNode)
    r2_cell = xydist2(node.p1, node.p2)
    y2_cell = xydist2(pi.y, value(node))
    (r2_cell / y2_cell) < state.θ^2
end

function bh_eligible(state::TSNEState, ni::QTNode, nj::QTNode)
    r2 = max(xydist2(ni.p1, ni.p2), xydist2(nj.p1, nj.p2))
    y2 = xydist2(value(ni), value(nj))
    (r2 / y2) < state.θ^2
end


function objective(state::TSNEState)
    ΣPij = state.ΣPij
    Z = state.Z
    kl = 0.0
    for (i, pi) ∈ state.points
        for (j, neighbor) ∈ pi.neighbors
            pj = state.points[j]
            pᵢⱼ = neighbor.pij / ΣPij
            if pᵢⱼ > 0.0
                qᵢⱼ = qij(pi, pj) / Z
                kl += pᵢⱼ * log(pᵢⱼ / qᵢⱼ)
            end
        end
    end
    kl
end

"""Gradient (Barnes-Hut)"""
function gradient(state::TSNEState, i::Sample, exaggeration)
    fₐ = f_attr(state, i) * exaggeration
    Z, fᵣ = f_rep(state, i, Z)
    4 * (fₐ - fᵣ)
end

update!(state::TSNEState) = update!(state, keys(state.points))

function update!(state::TSNEState, idxs)

    indices = collect(idxs)
    N = length(indices)
    T = Threads.nthreads()

    η = 500.0 # learning rate

    # update the forces    
    Z = state.Z = get_zterm(state)
    if Z <= 0.0
        Z = get_zterm(state)
        state.Z = Z
    end
    #println("Z = $Z !!!!!")
    #ΔZ = zeros(T * 10)
    @Threads.threads for x in 1:N
        i = indices[x]
        pi = state.points[i]
        oldzterm = pi.zterm
        fa = f_attr(state, i)
        (zterm, fr) = f_rep(state, i)
        pi.zterm = zterm
        #ΔZ[Threads.threadid() * 10] += zterm - oldzterm

        α = ifelse(pi.iterctr < 50, 0.5, 0.8)

        exaggeration = max(1.0, 4.0 - pi.iterctr * .01)

        # compute gradient
        ∂y = 4 * (fa * exaggeration - fr / Z)

        # update gain
        g1 = update_gain(pi.gain[:1], ∂y[:1], pi.Δy[:1])
        g2 = update_gain(pi.gain[:2], ∂y[:2], pi.Δy[:2])
        pi.gain = Point(g1, g2)

        # compute updated velocity
        pi.Δy = α * pi.Δy - η * (pi.gain .* ∂y)

    end
    #state.Z += sum(ΔZ)


    if N == count(state)
        # we're updating everything... just make a new quadtree
        r = 1000.0
        quadtree = make_quad_tree(Point(-r, -r), Point(r, r))
        for x in 1:N
            i = indices[x]
            pi = state.points[i]
            pi.y += pi.Δy
            pi.iterctr += 1
            insert!(quadtree, i, pi.y)
        end
        state.quadtree = quadtree
    else
        # otherwise, update quadtree in-place
        quadtree = state.quadtree
        for x in 1:N
            i = indices[x]
            pi = state.points[i]
            delete!(quadtree, i, pi.y)
            pi.y += pi.Δy
            pi.iterctr += 1
            insert!(quadtree, i, pi.y)
        end
    end

    nothing
end

update_gain(gain, dy, iy) = max(0.01, sign(dy) != sign(iy) ? gain + 0.2 : gain * 0.8)

const window_width = 2000
const window_height = 2000
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
const halocolor = SFML.Color(0,127,0)
const viewscale = 40

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

function view(window, state, labels; miniters::Int=0, newiters::Int=30)

    event = Event()

    while pollevent(window, event)
        if get_type(event) == EventType.CLOSED
            close(window)
            return false
        end
    end

    clear(window, SFML.white)

    for point in values(state.points)
        point.iterctr < miniters && continue

        xy = vec2screen(point.y)
        if point.iterctr < (miniters + newiters)
            halo = mkdot(xy[:1], xy[:2]; r=16)
            set_fillcolor(halo, halocolor)
            draw(window, halo)
        end

        dot = mkdot(xy[:1], xy[:2]; r=8, label=labels[point.id]+1)
        draw(window, dot)
    end

    display(window)
    true

end

function reposition!(state::TSNEState, i::Sample; miniters::Int=500)
    pi = state.points[i]
    avgy = Point(0.0, 0.0)
    ct = 0
    for (j, d) in closest(state.vptree, i, floor(state.perplexity))
        pj = state.points[j]
        if pj.iterctr > miniters
            ct += 1
            avgy += pj.y
        end
        ct == 3 && break
    end
    if ct >= 3
        avgy = avgy / ct
        delete!(state.quadtree, pi.id, pi.y)
        pi.y = avgy
        insert!(state.quadtree, pi.id, pi.y)
    end
    nothing
end

function run_beach_tsne(X, labels;
        init_size=100,
        window_size=1000,
        iterations=600,
        perplexity=20.0,
        θ=0.8,
        miniters=300,
        newiters=20,
        viewfps=30,
)

    D, N = size(X)

    # interface for fetching rows of X
    xrow = 0
    nextitem() = (xrow += 1; X[:,xrow])
    eof() = (xrow >= N)

    # construct TSNE state
    state = make_tsne_state(perplexity=perplexity, θ=θ)

    window = mkwindow()
    lastview = time()
    viewrate = 1. / viewfps

    function doview()
        t = time()
        res = true
        if (t - lastview) > viewrate
            res = view(window, state, labels; miniters=miniters, newiters=newiters)
            lastview = t
        end
        res
    end

    # load initial data
    @showprogress 1 "Inserting initial points" for i ∈ 1:init_size
        eof() && break
        insert!(state, nextitem())
    end

    # update initial neighborhoods
    @showprogress 1 "Computing point perplexities" for i ∈ 1:init_size
        eof() && break
        update_neighborhood!(state, i; neighbor_penalty=0.0)
    end

    # initial convergence
    @showprogress 1 "Initial convergence" for t ∈ 1:iterations
        update!(state)
        doview() || return
        if t % 20 == 0
            println("t=$t  count=$(count(state))  vpdepth=$(depth(state.vptree))  kl=$(objective(state))")
        end
    end

    # window cycling phase
    nnupdates = 0
    @showprogress 1 "Cycling window" for t=1:(N-init_size)

        while count(state) >= window_size
            remove!(state, 1 + xrow - window_size)
        end

        insert!(state, nextitem())
        update_neighborhood!(state, xrow)
        reposition!(state, xrow; miniters=200)

        # determine which points to update
        #update_pts = Vector{Int}()
        #logs = log(rand())
        for (i, point) ∈ state.points
            # update neighborhood
            if point.staleness > 3perplexity
               update_neighborhood!(state, point.id; neighbor_penalty=0.0)
               nnupdates += 1
            end
            #logp = -min(point.iterctr, iterations) / 0.5iterations
            #(logs < logp) && push!(update_pts, point.id)
        end

        #update!(state, update_pts)
        update!(state)

        if t % 20 == 0
            println("t=$t  count=$(count(state))  vpdepth=$(depth(state.vptree))  kl=$(objective(state))  nnupdates=$nnupdates")
        end


        doview() || return
    end

    # final convergence
    @showprogress 1 "Final convergence" for t ∈ 1:iterations

        if t % 20 == 0
            println("t=$t  count=$(count(state))  vpdepth=$(depth(state.vptree))  kl=$(objective(state))")
        end

        update!(state)
        doview() || return
    end

    nothing

end


function main()
    s = ArgParseSettings(description = "beach_tsne.jl")

    @add_arg_table s begin
        "infile"
        "--init-size"
            arg_type = Int
            default = 100
        "--window-size"
            arg_type = Int
            default = 1000
        "--iterations"
            arg_type = Int
            default = 600
        "--perplexity"
            arg_type = Float64
            default = 20.0
        "--theta"
            arg_type = Float64
            default = 0.8
        "--min-iters"
            arg_type = Int
            default = 300
        "--new-iters"
            arg_type = Int
            default = 20
    end

    args = parse_args(s)

    fname = args["infile"]
    X = map(Float64, h5read(fname, "X"))
    labels = h5read(fname, "labels")

    # normalize pixel values to 0..1
    X ./= 255.0

    run_beach_tsne(X, labels;
        init_size=args["init-size"],
        window_size=args["window-size"],
        iterations=args["iterations"],
        perplexity=args["perplexity"],
        miniters=args["min-iters"],
        newiters=args["new-iters"],
    )

end


main()


#end # module




