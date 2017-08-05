using VPTrees
using Base.Test


@testset "VPTree prebuild tests" begin

    # setup tree
    ids  = [1,   2,   3,   4,   5,   6,   7,   8,    9,    10]
    data = [0.5, 1.0, 3.0, 3.1, 4.5, 5.5, 6.5, 10.0, 11.5, 13.25]
    dist(i, j) = Float64(abs(data[i] - data[j]))
    tree = make_vp_tree(dist, ids)

    @test count(tree) == 10
    @test closest(tree, 8)[1] == 9
    @test closest(tree, 3)[1] == 4

    @test searchall(tree, 8, 4.0) == [(9, 1.5), (10, 3.25), (7, 3.5)]

    @test sort!(collect(tree)) == ids

end

@testset "VPTree insert tests" begin

    ids = Vector{Int}()
    data = Vector{Float64}()    
    dist(i, j) = Float64(abs(data[i] - data[j]))
    tree = make_vp_tree(dist, ids)

    @test count(tree) == 0

    # setup tree
    add_ids  = [1,   2,   3,   4,   5,   6,   7,   8,    9,    10]
    add_data = [0.5, 1.0, 3.0, 3.1, 4.5, 5.5, 6.5, 10.0, 11.5, 13.25]
    N = length(add_ids)

    for i in 1:N
        push!(ids, add_ids[i])
        push!(data, add_data[i])
        insert!(tree, add_ids[i])
    end

    @test count(tree) == 10

    # test search functionality
    @test closest(tree, 8)[1] == 9
    @test closest(tree, 3)[1] == 4
    @test searchall(tree, 8, 4.0) == [(9, 1.5), (10, 3.25), (7, 3.5)]

    # test collect functionality
    @test sort!(collect(tree)) == add_ids

    for i in [1, 3, 5, 7, 9]
        remove!(tree, i)
    end
    @test sort!(collect(tree)) == [2, 4, 6, 8, 10]

end

@testset "Torture" begin

    data = collect(1:10000)
    dist(i, j) = Float64(abs(data[i] - data[j]))

    println("Building 10000 nodes")
    @time tree = make_vp_tree(dist, data)

    @test count(tree) == 10000

    println("depth = $(depth(tree))")

    stuff = searchall(tree, 50, 5.5)
    @test length(stuff) == 10

    stuff = collect(10001:20000)
    shuffle!(stuff)

    println("Inserting + Delete 10000 nodes")
    for i in 10001:20000
        push!(data, i)
    end
    @time for i in stuff
        remove!(tree, i - 10000)
        @assert count(tree) == 9999
        insert!(tree, i)
        @assert count(tree) == 10000
    end

    @test count(tree) == 10000

    #stuff = searchall(tree, 150, 5.5)
    #@test length(stuff) == 10

    println("depth = $(depth(tree))")
    println("closest to 15000 = $(closest(tree, 15000))")

    # println("Deleting 10000 nodes (every other sample)")
    # @time for i in 1:2:20000
    #     remove!(tree, i)
    # end

    # @test sort!(collect(tree)) == collect(2:2:20000)
    # println("depth = $(depth(tree))")

    g = garbage!(tree)
    println("garbage = $(length(g))")

end


# test()
