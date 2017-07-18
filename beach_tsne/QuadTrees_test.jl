using QuadTrees
using Base.Test


@testset "QuadTree insert / delete tests" begin

    const N = 10000

    p1 = Point(-10, -10)
    p2 = Point(10, 10)
    points = Vector{Point}()
    tree = make_quad_tree(p1, p2)

    @time for i in 1:N
        x = rand() * 20.0 - 10.0
        y = rand() * 20.0 - 10.0
        p = Point(x, y)
        push!(points, p)
        insert!(tree, i, p)
    end

    @test count(tree) == N

    println("center of mass: $(value(tree.root))")

    @time for i in 1:N
        println(i)
        delete!(tree, i, points[i])
    end

    @test count(tree) == 0

end
