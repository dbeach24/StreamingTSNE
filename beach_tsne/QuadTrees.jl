module QuadTrees

using StaticArrays

const Point = SVector{2, Float64}

import Base: insert!, count

export QuadTree, make_quad_tree, Point, count, center, insert!

immutable QTNode
    p1::Point
    p2::Point
    id::Int
    value::Point
    count::Int

    # children
    ul::Nullable{QTNode}
    ur::Nullable{QTNode}
    ll::Nullable{QTNode}
    lr::Nullable{QTNode}
end

const QTNodePtr = Nullable{QTNode}

value(node::QTNode) = node.value
value(node::QTNodePtr) = isnull(node) ? Point(0.0, 0.0) : value(get(node))
count(node::QTNode) = node.count
count(node::QTNodePtr) = isnull(node) ? 0 : count(get(node))
center(node::QTNode) = (node.p1 .+ node.p2) ./ 2
center(node::QTNodePtr) = isnull(node) ? Point(0.0, 0.0) : center(get(node))

function quadrant(node::QTNode, value::Point)
    ctr = center(node)
    isleft = node.p1[:1] <= value[:1] < ctr[:1]
    islower = node.p1[:2] <= value[:2] < ctr[:2]
    (isleft, islower)
end

function make_node(p1::Point, p2::Point, id::Int, value::Point)
    @assert p1[:1] < p2[:1] && p1[:2] < p2[:2]
    QTNodePtr(QTNode(p1, p2, id, value, 1, nothing, nothing, nothing, nothing))
end

function compose_node(p1::Point, p2::Point, ul::QTNodePtr, ur::QTNodePtr, ll::QTNodePtr, lr::QTNodePtr)
    # TODO sanity checks?

    # compute center of mass from children and build node
    total = count(ul) + count(ur) + count(ll) + count(lr)
    wul = count(ul) / total
    wur = count(ur) / total
    wll = count(ll) / total
    wlr = count(lr) / total
    center = value(ul) .* wul .+ value(ur) .* wur .+ value(ll) .* wll .+ value(lr) .* wlr

    QTNodePtr(QTNode(p1, p2, -1, center, total, ul, ur, ll, lr))
end

make_ll_child(p1, ctr, p2, id, value) = make_node(p1, ctr, id, value)
make_ul_child(p1, ctr, p2, id, value) = make_node(Point(p1[:1], ctr[:2]), Point(ctr[:1], p2[:2]), id, value)
make_lr_child(p1, ctr, p2, id, value) = make_node(Point(ctr[:1], p1[:2]), Point(p2[:1], ctr[:2]), id, value)
make_ur_child(p1, ctr, p2, id, value) = make_node(ctr, p2, id, value)

function insert(node::QTNode, id::Int, value::Point)
    p1 = node.p1
    p2 = node.p2

    @assert p1[:1] <= value[:1] < p2[:1]
    @assert p1[:2] <= value[:2] < p2[:2]

    ul = node.ul
    ur = node.ur
    ll = node.ll
    lr = node.lr

    ctr = center(node)

    if count(node) == 1
        # this is a leaf node which must become a parent
        # move this node's data into the appropriate child
        nid = node.id
        nvalue = node.value
        isleft, islower = quadrant(node, nvalue)
        if isleft
            if islower
                ll = isnull(ll) ? make_ll_child(p1, ctr, p2, nid, nvalue) : insert(get(ll), nid, nvalue)
            else
                ul = isnull(ul) ? make_ul_child(p1, ctr, p2, nid, nvalue) : insert(get(ul), nid, nvalue)
            end
        else
            if islower
                lr = isnull(lr) ? make_lr_child(p1, ctr, p2, nid, nvalue) : insert(get(lr), nid, nvalue)
            else
                ur = isnull(ur) ? make_ur_child(p1, ctr, p2, nid, nvalue) : insert(get(ur), nid, nvalue)
            end
        end
    end

    # now, add/insert the new value into the appropriate quadrant of this node
    isleft, islower = quadrant(node, value)

    if isleft
        if islower
            ll = isnull(ll) ? make_ll_child(p1, ctr, p2, id, value) : insert(get(ll), id, value)
        else
            ul = isnull(ul) ? make_ul_child(p1, ctr, p2, id, value) : insert(get(ul), id, value)
        end
    else
        if islower
            lr = isnull(lr) ? make_lr_child(p1, ctr, p2, id, value) : insert(get(lr), id, value)
        else
            ur = isnull(ur) ? make_ur_child(p1, ctr, p2, id, value) : insert(get(ur), id, value)
        end
    end

    compose_node(p1, p2, ul, ur, ll, lr)

end


function delete(node::QTNode, id::Int, value::Point)
    p1 = node.p1
    p2 = node.p2
end



type QuadTree
    p1::Point
    p2::Point
    root::Nullable{QTNode}
end

make_quad_tree(p1::Point, p2::Point) = QuadTree(p1, p2, nothing)

count(tree::QuadTree) = count(tree.root)

function insert!(tree::QuadTree, id::Int, value::Point)
    if isnull(tree.root)
        tree.root = make_node(tree.p1, tree.p2, id, value)
    else
        tree.root = insert(get(tree.root), id, value)
    end
end






end # module