#!/usr/bin/env julia

using ProgressMeter
using HDF5
using ArgParse

#using Gadfly

# Numpy Math.sum => axis = 0 => sum down the columns. axis = 1 => sum along the rows
# Julia Base.sum => axis = 1 => sum down the columns. axis = 2 => sum along the rows

#
#  tsne.jl
#
#  This is a straight off Julia port of Laurens van der Maatens python implementation of tsne

"""
Compute the point perplexities `P` given its distances to the other points `D`
and the precision of Gaussian distribution `beta`.
"""
function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @simd for j in eachindex(D)
        @inbounds P[j] = exp(-beta * D[j])
    end
    sumP = sum(P)
    @assert (isfinite(sumP) && sumP > 0.0) "Degenerated P: sum=$sumP, beta=$beta"
    H = log(sumP) + beta * dot(D, P) / sumP
    @assert isfinite(H) "Degenerated H"
    scale!(P, 1/sumP)
    return H
end

"""
    perplexities(X::Matrix, tol::Number = 1e-5, perplexity::Number = 30.0;
                 [keyword arguments])

Convert `n×d` matrix `X` of point coordinates into `n×n` perplexities matrix `P`.
Performs a binary search to get P-values in such a way that each conditional
Gaussian has the same perplexity.
"""
function perplexities(X::Matrix, tol::Number = 1e-5, perplexity::Number = 30.0;
                      max_iter::Integer = 50,
                      verbose::Bool=false, progress::Bool=true)
    verbose && info("Computing pairwise distances...")
    (n, d) = size(X)
    sum_XX = sum(abs2, X, 2)
    D = -2 * (X*X') .+ sum_XX .+ sum_XX' # euclidean distances between the points
    @show sum(X[1,:])
    @show sum(X[2,:])
    @show D[1,1]
    @show D[1,2]
    @show D[2,1]
    P = zeros(Float64, n, n) # perplexities matrix
    beta = ones(Float64, n)  # vector of Normal distribution precisions for each point
    logU = log(perplexity) # the log of expected perplexity
    Di = zeros(Float64, n)
    Pcol = zeros(Float64, n)

    # Loop over all datapoints
    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && update!(pb, i)

        # Compute the Gaussian kernel and entropy for the current precision
        betai = 1.0
        betamin = 0.0
        betamax = Inf

        copy!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf) # exclude D[i,i] from minimum(), yet make it finite and exp(-D[i,i])==0.0

        H = Hbeta!(Pcol, Di, betai)
        Hdiff = H - logU

        # Evaluate whether the perplexity is within tolerance
        tries = 0
        while abs(Hdiff) > tol && tries < max_iter
            # If not, increase or decrease precision
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            # Recompute the values
            H = Hbeta!(Pcol, Di, betai)
            Hdiff = H - logU
            tries += 1
        end
        verbose && abs(Hdiff) > tol && warn("P[$i]: perplexity error is above tolerance: $(Hdiff)")
        # Set the final column of P
        @assert Pcol[i] == 0.0 "Diagonal probability P[$i,$i]=$(Pcol[i]) not zero"
        P[:, i] = Pcol
        beta[i] = betai
    end
    println("Beta = $(beta)")
    progress && finish!(pb)
    # Return final P-matrix
    verbose && info("Mean σ=$(mean(sqrt.(1 ./ beta)))")
    return P
end

"""
    pca(X::Matrix, ncols::Integer = 50)

Run PCA on `X` to reduce the number of its dimensions to `ndims`.

FIXME use PCA routine from JuliaStats?
"""
function pca{T}(X::Matrix{T}, ndims::Integer = 50)
    (n, d) = size(X)
    X = X - repmat(mean(X, 1), n, 1)
    C = (X' * X) ./ (size(X,1)-1)
    l, M = eig(C)
    sorder = sortperm(l, rev=true)
    M = M[:, sorder]::Matrix{T}
    Y = X * M[:, 1:min(d, ndims)]
    return Y
end

# K-L divergence element
kldivel(p, q) = (@fastmath t = ifelse(p > zero(p) && q > zero(q), p*log(p/q), zero(p)); t)

"""
    tsne(X::Matrix, ndims::Integer=2, reduce_dims::Integer=0,
         max_iter::Integer=1000, perplexity::Number=30.0; [keyword arguments])

Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) to `X`,
i.e. embed its points (rows) into `ndims` dimensions preserving close neighbours.

Different from the orginal implementation,
the default is not to use PCA for initialization.

### Arguments
* `reduce_dims` the number of the first dimensions of `X` PCA to use for t-SNE,
  if 0, all available dimension are used
* `pca_init` whether to use the first `ndims` of `X` PCA as the initial t-SNE layout,
  if `false` (the default), the method is initialized with the random layout
* `max_iter` how many iterations of t-SNE to do
* `perplexity` the number of "effective neighbours" of a datapoint,
  typical values are from 5 to 50, the default is 30
* `verbose` output informational and diagnostic messages
* `progress` display progress meter during t-SNE optimization
* `min_gain`, `eta`, `initial_momentum`, `final_momentum`, `momentum_switch_iter`,
  `stop_cheat_iter`, `cheat_scale` low-level parameters of t-SNE optimization

See also [Original t-SNE implementation](https://lvdmaaten.github.io/tsne).
"""
function tsne(X::Matrix, ndims::Integer = 2, reduce_dims::Integer = 0,
              max_iter::Integer = 1000, perplexity::Number = 30.0;
              min_gain::Number = 0.01, eta::Number = 200.0, pca_init::Bool = false,
              initial_momentum::Number = 0.5, final_momentum::Number = 0.8, momentum_switch_iter::Integer = 250,
              stop_cheat_iter::Integer = 250, cheat_scale::Number = 12.0,
              verbose::Bool = false, progress::Bool=true)

    start_time = time()

    verbose && info("Initial X Shape is $(size(X))")
    ndims < size(X, 2) || throw(DimensionMismatch("X has fewer dimensions ($(size(X,2))) than ndims=$ndims"))

    # Initialize variables
    T = eltype(X)
    X ./= 255.0
    #X = X * (one(T)/(std(X)::T)) # note that X is copied
    # if 0<reduce_dims<size(X, 2)
    #     reduce_dims = max(reduce_dims, ndims)
    #     verbose && info("Preprocessing the data using PCA...")
    #     X = pca(X, reduce_dims)
    # end
    (n, d) = size(X)
    if !pca_init
        verbose && info("Starting with random layout...")
        Y = randn(n, ndims)
    else
        verbose && info("Using the first $ndims components of the data PCA as the initial layout...")
        if reduce_dims >= ndims
            Y = X[:, 1:ndims] # reuse X PCA
        else
            @assert reduce_dims <= 0 # no X PCA
            Y = pca(X, ndims)
        end
    end

    dY = zeros(T, n, ndims)
    iY = zeros(T, n, ndims)
    gains = ones(T, n, ndims)

    # Compute P-values
    P = perplexities(X, 1e-5, perplexity, verbose=verbose, progress=progress)::Matrix{T}
    P = P + P' # make P symmetric
    scale!(P, 1.0/sum(P))
    scale!(P, cheat_scale)  # early exaggeration
    sum_P = cheat_scale
    L = zeros(P)
    Ymean = zeros(T, 1, ndims)
    sum_YY = zeros(T, n, 1)
    Lcolsums = zeros(T, n, 1)
    last_kldiv = NaN

    # Run iterations
    progress && (pb = Progress(max_iter, "Computing t-SNE"))
    Q = zeros(P)
    for iter in 1:max_iter
        # Compute pairwise affinities
        sum!(abs2, sum_YY, Y)
        BLAS.syrk!('U', 'N', 1.0, Y, 0.0, Q) # Q=YY^T, updates only the upper tri of Q
        @inbounds for j in 1:size(Q, 2)
            sum_YYj_p1 = 1.0 + sum_YY[j]
            Qj = view(Q, :, j)
            Qj[j] = 0.0
            @simd for i in 1:(j-1)
                denom = sum_YYj_p1 - 2.0 * Qj[i] + sum_YY[i]
                @fastmath Qj[i] = ifelse(denom > 1.0, 1.0 / denom, 1.0)
            end
        end
        sum_Q = 2*sum(Q) # the diagonal and lower-tri part of Q is zero

        # Compute the gradient
        inv_sumQ = 1/sum_Q
        fill!(Lcolsums, zero(T)) # column sums
        # fill the upper triangle of L and P
        @inbounds for j in 1:size(L, 2)
            Lj = view(L, :, j)
            Pj = view(P, :, j)
            Qj = view(Q, :, j)
            Lsumj = zero(T)
            @simd for i in 1:j
                Lj[i] = l = (Pj[i] - Qj[i]*inv_sumQ) * Qj[i]
                Lcolsums[i] += l
                Lsumj += l
            end
            Lcolsums[j] += Lsumj - Lj[j]
        end
        @inbounds for (i, ldiag) in enumerate(Lcolsums)
            L[i, i] -= ldiag
        end
        # dY = -4LY
        BLAS.symm!('L', 'U', -4.0, L, Y, zero(T), dY)

        # Perform the update
        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds @simd for i in eachindex(gains)
            gains[i] = max(ifelse((dY[i] > 0) == (iY[i] > 0),
                                  gains[i] * 0.8,
                                  gains[i] + 0.2),
                           min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        mean!(Ymean, Y)
        @inbounds for j in 1:size(Y, 2)
            YcolMean = Ymean[j]
            Yj = view(Y, :, j)
            @simd for i in eachindex(Yj)
                Yj[i] -= YcolMean
            end
        end

        # Compute current value of cost function
        if progress && (!isfinite(last_kldiv) || iter == max_iter || mod(iter, max(max_iter÷20, 10)) == 0)
            local kldiv = 0.0
            @inbounds for j in 1:size(P, 2)
                Pj = view(P, :, j)
                Qj = view(Q, :, j)
                kldiv_j = 0.0
                @simd for i in 1:(j-1)
                    # P and Q are symmetric (only the upper triangle used)
                    kldiv_j += kldivel(Pj[i], Qj[i])
                end
                kldiv += 2*kldiv_j + kldivel(Pj[j], Q[j])
            end
            last_kldiv = kldiv/sum_P + log(sum_Q/sum_P) # adjust wrt P and Q scales
        end
        progress && update!(pb, iter, showvalues = Dict(:KL_divergence => last_kldiv))
        # stop cheating with P-values
        if iter == min(max_iter, stop_cheat_iter)
            scale!(P, 1/sum_P)
            sum_P = 1.0
        end
    end
    progress && (finish!(pb))
    verbose && info("Final t-SNE KL-divergence=$last_kldiv")

    elapsed_time = time() - start_time
    verbose && info("Elaspsed time=$elapsed_time")

    # Return solution
    return Y, last_kldiv
end


"""
Normalize `A` columns, so that the mean and standard deviation
of each column are 0 and 1, resp.
"""
function rescale(A, dim::Integer=1)
    res = A .- mean(A, dim)
    dev = std(A, dim)
    res ./= map!(x -> x > 0.0 ? x : 1.0, dev, dev)
    res
end

function main_test(infile, outfile; initial_dims=50, iterations=1000, perplexity=20)

    X = map(Float64, h5read(infile, "X")')
    labels = h5read(infile, "labels")

    println("X dimensions are: ", size(X))
    start = time()
    Y, last_kldiv = tsne(X, 2, initial_dims, iterations, perplexity, verbose=true)
    stop = time()
    println("Y dimensions are: ", size(Y))

    h5open(outfile, "w") do file
        write(file, "Y", Y')
        write(file, "labels", labels)
        attrs(file)["algo"] = "Julia N^2"
        attrs(file)["input"] = infile
        attrs(file)["time"] = (stop - start)
        attrs(file)["error"] = last_kldiv
    end

    # compensate for broken Python HDF5
    # (by writing string attributes using Python!)
    run(`./h5writeattr.py $(outfile) "algo=Julia N^2 Mod" "input=$(infile)"`)

end

function main()
    s = ArgParseSettings(description = "tsne_julia.jl")

    @add_arg_table s begin
        "infile"
        "outfile"
        "--initial_dims"
            arg_type = Int
            default = 50
        "--iterations"
            arg_type = Int
            default = 1000
        "--perplexity"
            arg_type = Int
            default = 20
    end

    args = parse_args(s) # the result is a Dict{String,Any}

    main_test(args["infile"], args["outfile"],
        initial_dims=args["initial_dims"],
        iterations=args["iterations"],
        perplexity=args["perplexity"]
    )

end

main()


