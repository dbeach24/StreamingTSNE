using QuadTrees
using Base.Test


@testset "QuadTree insert tests" begin

    p1 = Point(-10, -10)
    p2 = Point(10, 10)
    tree = make_quad_tree(p1, p2)

    for i in 1:100
        x = rand() * 20.0 - 10.0
        y = rand() * 20.0 - 10.0
        p = Point(x, y)
        insert!(tree, i, p)
    end

    @test count(tree) == 100

end
