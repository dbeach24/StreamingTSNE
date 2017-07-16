module VPTrees

import Base: search, insert!, count, collect

export Sample, Radius
export VPNode, VPTree
export eachkey, eachnode
export count, depth, eachkey, eachnode, collect
export make_vp_tree, insert!, search, searchall, closest

const Sample = Int
const Radius = Float64

const REBALANCE_DEPTH = 5

immutable NodeStats
    count::Int
    depth::Int
end

immutable VPNode
    p::Sample
    μ::Radius
    stats::NodeStats
    left::Nullable{VPNode}
    right::Nullable{VPNode}
end

const VPNodePtr = Nullable{VPNode}

type VPTree
    dist::Function
    root::VPNodePtr
end

stats() = NodeStats(1, 1)
stats(n::VPNode) = n.stats
stats(n::VPNodePtr) = isnull(n) ? NodeStats(0, 0) : stats(get(n))
stats(l::NodeStats, r::NodeStats) = NodeStats(1 + count(l) + count(r), 1 + max(depth(l), depth(r)))
stats(l::VPNodePtr, r::VPNodePtr) = stats(stats(l), stats(r))

"""
    count(tree|node)

Return the total number of samples in a given tree or subtree.
"""
count(t::VPTree) = count(t.root)
count(n::VPNode) = count(n.stats)
count(n::VPNodePtr) = isnull(n) ? 0 : count(get(n))
count(s::NodeStats) = s.count

"""
    depth(tree|node)

Return the depth of the given tree or subtree.
"""
depth(t::VPTree) = depth(t.root)
depth(n::VPNode) = depth(n.stats)
depth(n::VPNodePtr) = isnull(n) ? 0 : depth(get(n))
depth(s::NodeStats) = s.depth

"""
    eachkey(f, tree|node)
    eachnode(f, tree|node)

Generically treverse tree or subtree, applying `f(n)` to each key or node.
"""
eachkey(f::Function, t::VPTree) = eachnode(x -> f(x.p), t)
eachkey(f::Function, n::VPNodePtr) = eachnode(x -> f(x.p), n)
eachnode(f::Function, t::VPTree) = eachnode(f, t.root)
eachnode(f::Function, n::VPNodePtr) = !isnull(n) ? eachnode(f, get(n)) : nothing
function eachnode(f::Function, t::VPNode)
    f(t)
    eachnode(f, t.left)
    eachnode(f, t.right)
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

make_vp_tree(dist::Function, S::Vector{Sample}=Vector{Sample}()) = VPTree(dist, make_vp_node(S, dist))

make_vp_node(p::Sample, μ::Radius, left::VPNodePtr, right::VPNodePtr) = VPNode(p, μ, stats(left, right), left, right)

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

function insert!(t::VPTree, s::Sample)
    t.root = insert(t.root, s, t.dist)
    t
end

function insert(n::VPNodePtr, s::Sample, dist::Function) :: VPNodePtr
    isnull(n) && return make_vp_node(s, 0.0, VPNodePtr(), VPNodePtr())

    node = get(n)

    x = dist(node.p, s)

    if count(node) == 1
        # this is a leaf node, set radius and add as right child
        make_vp_node(node.p, x, node.left, insert(node.right, s, dist))
    else
        # this is a non-leaf node... navigate to appropriate child
        left = node.left
        right = node.right
        if x < node.μ
            left = insert(node.left, s, dist)
        else
            right = insert(node.right, s, dist)
        end
        make_vp_node(node.p, node.μ, left, right)
    end
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
    x = dist(q, n.p)
    if x < τ
        τ = f(n.p, x)
    end
    τ < 0 && return τ
    if x - τ < n.μ
        τ = search(f, n.left, q, τ, dist)
        τ < 0 && return τ
    end
    if x + τ >= n.μ
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

function verify(t::VPTree)
    eachnode(t) do node
        if !isnull(t.left)
            nothing
        end
        if !isnull(t.right)
            nothing
        end
    end
end

end # module

