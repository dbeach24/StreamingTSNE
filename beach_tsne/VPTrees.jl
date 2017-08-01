module VPTrees

import Base: search, insert!, count, collect, delete!

using DataStructures

export Sample, Radius
export VPNode, VPTree
export eachkey, eachnode
export count, depth, eachkey, eachnode, collect, garbage!
export make_vp_tree, insert!, search, searchall, closest, delete!

const Sample = Int
const Radius = Float64

# Parameters to control the partial auto-rebalancing
const REBALANCE_DEPTH = 5
const MIN_COUNT = 12

immutable NodeStats
    total::Int
    depth::Int
    ndel::Int
end

immutable VPNode
    p::Sample
    μ::Radius
    stats::NodeStats
    left::Nullable{VPNode}
    right::Nullable{VPNode}
    active::Bool
end

const VPNodePtr = Nullable{VPNode}

type VPTree
    dist::Function
    root::VPNodePtr
    garbage::Vector{Sample}
end

stats() = NodeStats(1, 1, 0)
stats(n::VPNode) = n.stats
stats(n::VPNodePtr) = isnull(n) ? NodeStats(0, 0, 0) : stats(get(n))
stats(l::NodeStats, r::NodeStats, active::Bool) = NodeStats(
    1 + size(l) + size(r),
    1 + max(depth(l), depth(r)),
    ndel(l) + ndel(r) + (!active ? 1 : 0)
)
stats(l::VPNodePtr, r::VPNodePtr, active::Bool) = stats(stats(l), stats(r), active)

size(t::VPTree) = size(t.root)
size(n::VPNode) = size(n.stats)
size(n::VPNodePtr) = isnull(n) ? 0 : size(get(n))
size(s::NodeStats) = s.total

"""
    count(tree|node)

Return the total number of active samples in a given tree or subtree.
"""
count(t::VPTree) = count(t.root)
count(n::VPNode) = count(n.stats)
count(n::VPNodePtr) = isnull(n) ? 0 : count(get(n))
count(s::NodeStats) = s.total - s.ndel

"""
    depth(tree|node)

Return the depth of the given tree or subtree.
"""
depth(t::VPTree) = depth(t.root)
depth(n::VPNode) = depth(n.stats)
depth(n::VPNodePtr) = isnull(n) ? 0 : depth(get(n))
depth(s::NodeStats) = s.depth

ndel(t::VPTree) = ndel(t.root)
ndel(n::VPNode) = ndel(n.stats)
ndel(n::VPNodePtr) = isnull(n) ? 0 : ndel(get(n))
ndel(s::NodeStats) = s.ndel

"""
    eachkey(f, tree|node)
    eachnode(f, tree|node)

Generically treverse tree or subtree, applying `f(n)` to each key or node.
"""
eachkey(f::Function, t::VPTree) = eachnode(x -> x.active && f(x.p), t)
eachkey(f::Function, n::VPNodePtr) = eachnode(x -> x.active && f(x.p), n)
eachnode(f::Function, t::VPTree) = eachnode(f, t.root)
eachnode(f::Function, n::VPNodePtr) = !isnull(n) ? eachnode(f, get(n)) : nothing
function eachnode(f::Function, n::VPNode)
    f(n)
    eachnode(f, n.left)
    eachnode(f, n.right)
    nothing
end

"""
    collect(tree|node)

Return all samples from a tree or subtree as a vector.
"""
collect(t::VPTree) = collect(t.root)
function collect(n::VPNodePtr)
    result = Vector{Sample}()
    eachkey(n) do s
        push!(result, s)
    end
    result
end

make_vp_tree(dist::Function, S::Vector{Sample}=Vector{Sample}()) = VPTree(dist, make_vp_node(S, dist), Vector{Sample}())

make_vp_node(p::Sample, μ::Radius, left::VPNodePtr, right::VPNodePtr, active::Bool=true) = VPNode(p, μ, stats(left, right, active), left, right, active)

function make_vp_node(S::Vector{Sample}, dist::Function) :: VPNodePtr
    isempty(S) && return nothing

    p = select_vp(S, dist)
    dists = [dist(p, s) for s ∈ S]
    μ = median(dists)
    count = length(S)
    L = Vector{Sample}()
    R = Vector{Sample}()
    for (s, d) ∈ zip(S, dists)
        s == p && continue
        push!((d < μ) ? L : R, s)
    end

    left = make_vp_node(L, dist)
    right = make_vp_node(R, dist)

    make_vp_node(p, μ, left, right)
end

select_vp(S, dist) = rand(S)

function garbage!(t::VPTree)
    old_garbage = t.garbage
    t.garbage = Vector{Sample}()
    old_garbage
end

function rebuild(t::VPTree)
    t.root = rebuild(t.root, t.garbage, t.dist)
end

rebuild(n::VPNodePtr, garbage::Vector{Sample}, dist::Function) = make_vp_node(collect_garbage(n, garbage), dist)


function insert!(t::VPTree, s::Sample)
    t.root = insert(t.root, s, t.garbage, t.dist)
    t
end

function insert(n::VPNodePtr, s::Sample, garbage::Vector{Sample}, dist::Function) :: VPNodePtr
    isnull(n) && return make_vp_node(s, 0.0, VPNodePtr(), VPNodePtr())

    node = get(n)

    x = dist(node.p, s)

    if size(node) == 1
        # this is a leaf node, set radius and add as right child
        make_vp_node(node.p, x, node.left, insert(node.right, s, garbage, dist), node.active)
    else
        # this is a non-leaf node... navigate to appropriate child
        left = node.left
        right = node.right
        if x < node.μ
            left = insert(node.left, s, garbage, dist)
        else
            right = insert(node.right, s, garbage, dist)
        end
        check_rebuild(VPNodePtr(make_vp_node(node.p, node.μ, left, right, node.active)), garbage, dist)
    end
end

function delete!(t::VPTree, q::Sample)
    t.root = delete!(t.root, q, t.garbage, t.dist)
    t
end

delete!(n::VPNodePtr, q::Sample, garbage::Vector{Sample}, dist::Function) :: VPNodePtr = !isnull(n) ? delete!(get(n), q, garbage, dist) : nothing
function delete!(n::VPNode, q::Sample, garbage::Vector{Sample}, dist::Function) :: VPNodePtr
    l = n.left
    r = n.right
    if n.p == q
        if isnull(l) && isnull(r)
            # this is a leaf node -- delete immediately
            push!(garbage, n.p)
            return nothing
        end
        return check_rebuild(VPNodePtr(make_vp_node(n.p, n.μ, l, r, false)), garbage, dist)
    end
    x = dist(n.p, q)
    if x < n.μ
        l = delete!(l, q, garbage, dist)
    else
        r = delete!(r, q, garbage, dist)
    end
    check_rebuild(VPNodePtr(make_vp_node(n.p, n.μ, l, r, n.active)), garbage, dist)
end


function check_rebuild(n::VPNodePtr, garbage::Vector{Sample}, dist::Function)
    if needs_rebuild(n)
        rebuild(n, garbage, dist)
    else
        n
    end
end

needs_rebuild(n::VPNodePtr) = !isnull(n) ? needs_rebuild(get(n)) : false

function collect_garbage(n::VPNodePtr, garbage::Vector{Sample})
    samples = Vector{Sample}()
    eachnode(n) do node
        if node.active
            push!(samples, node.p)
        else
            push!(garbage, node.p)
        end
        nothing
    end
    samples
end

function needs_rebuild(n::VPNode)
    depth(n) == REBALANCE_DEPTH && count(n) < MIN_COUNT
end


function search(f::Function, t::VPTree, q::Sample, τ::Radius)
    search(f, t.root, q, τ, t.dist)
end

function search(
    f::Function,
    node::VPNodePtr,
    q::Sample,
    τ::Radius,
    dist::Function
)
    isnull(node) && return τ

    n = get(node)
    x = dist(n.p, q)
    if n.active && x < τ
        # include this node in the search
        τ = f(n.p, x)
    end
    τ < 0 && return τ
    if x - τ < n.μ
        # search left (inside)
        τ = search(f, n.left, q, τ, dist)
        τ < 0 && return τ
    end
    if x + τ >= n.μ
        # search right (outside)
        τ = search(f, n.right, q, τ, dist)
    end
    τ
end

function searchall(tree::VPTree, q::Sample, τ::Radius)
    results = Vector{Tuple{Sample, Radius}}()
    search(tree, q, τ) do p, d
        p != q && push!(results, (p, d))
        τ
    end
    sort!(results, by=x -> x[:2])
    results
end

function closest(tree::VPTree, q::Sample)
    best_p = 0
    best_d = Inf
    search(tree, q, best_d) do p, d
        if p != q && d < best_d
            best_p = p
            best_d = d
        end
        best_d
    end
    (best_p, best_d)
end

function closest(tree::VPTree, q::Sample, k)
    pq = PriorityQueue{Sample, Radius, Base.Order.ForwardOrdering}()
    τ = Inf
    search(tree, q, τ) do p, d
        if p != q
            enqueue!(pq, p, -d)
            if length(pq) > k
                dequeue!(pq)
                τ = -(peek(pq)[:2])
            end
        end
        τ
    end
    results = Vector{Tuple{Sample, Radius}}()
    while !isempty(pq)
        (p, nd) = peek(pq)
        push!(results, (p, -nd))
        dequeue!(pq)
    end
    reverse!(results)
    results
end


end # module

