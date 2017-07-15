#!/usr/bin/env python

from time import time
import argparse

import h5py
import numpy as np

from sklearn import manifold, datasets


def run(in_name, out_name, n_components=2, perplexity=20, iterations=1000, method='barnes_hut'):

    f = h5py.File(in_name, "r")
    X = f["X"].value
    n_samples = X.shape[1]
    labels = f["labels"].value

    start = time()
    tsne = manifold.TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=iterations,
        init="pca",
        random_state=0,
        method=method
    )
    Y = tsne.fit_transform(X)
    stop = time()

    error = tsne.kl_divergence_

    print("t-SNE: %.2g sec" % (stop - start))
    print("error: %.2g" % error)

    outf = h5py.File(out_name, "w")
    outf["Y"] = Y
    outf["labels"] = labels
    outf.attrs["algo"] = "Scikit Barnes-Hut" if method == 'barnes_hut' else "Scikit Exact"
    outf.attrs["input"] = in_name
    outf.attrs["time"] = (stop - start)
    outf.attrs["error"] = error


def main():

    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add("infile", type=str)
    add("outfile", type=str)
    add("--iterations", type=int, default=1000)
    add("--perplexity", type=int, default=20)
    add("--method", type=str, default="barnes_hut")

    args = parser.parse_args()
    run(args.infile, args.outfile,
        iterations=args.iterations,
        perplexity=args.perplexity,
        method=args.method
    )


if __name__ == "__main__":
    main()


