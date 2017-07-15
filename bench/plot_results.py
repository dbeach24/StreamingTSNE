#!/usr/bin/env python

import sys

import numpy as np
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter


def build_plot(fname, ax):

    f = h5py.File(fname, "r")
    Y = f["Y"].value
    labels = f["labels"].value


    #algo = "foo"
    #infile = "bar"
    algo = f.attrs["algo"]
    infile = f.attrs["input"]
    N = Y.shape[0]
    time = f.attrs["time"]
    error = f.attrs["error"]

    title = "%s on %s\nN=%d, time=%5.1f s; kl=%.4g" % (algo, infile, N, time, error)

    plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.title(title)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


def show_plots(fnames):

    N = len(fnames)
    cols = 2
    rows = (N + cols - 1) // cols

    fig = plt.figure(figsize=(10, 10))

    for i, fname in enumerate(fnames, 1):

        ax = fig.add_subplot(rows, cols, i)
        build_plot(fname, ax)

    plt.show()


if __name__ == "__main__":
    show_plots(sys.argv[1:])

