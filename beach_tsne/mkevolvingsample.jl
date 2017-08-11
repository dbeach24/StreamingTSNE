#!/usr/bin/env julia5


using ArgParse
using HDF5
using Distributions


function main()
    s = ArgParseSettings(description = "mkevolvingsample.jl")

    @add_arg_table s begin
        "infile"
        "outfile"
        "--stdfrac"
            arg_type = Float64
            default = 0.2
    end

    args = parse_args(s)

    infile = args["infile"]
    outfile = args["outfile"]
    stdfrac = args["stdfrac"]

    X = h5read(infile, "X")
    labels = h5read(infile, "labels")

    D, N = size(X)

    jumbler = Normal(0.0, stdfrac*N)

    p = sortperm(labels)

    j = [i + rand(jumbler) for i in 1:N]
    jp = sortperm(j)

    #pp = jp[p]
    pp = p[jp]

    Xpp = X[:,pp]
    labelspp = labels[pp]

    h5open(outfile, "w") do file
        write(file, "X", Xpp)
        write(file, "labels", labelspp)
    end

end


main()