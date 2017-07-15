#!/usr/bin/env python

import sys
import numpy
import h5py
import matplotlib.pyplot as plt


def main():

    data = []

    for fname in sys.argv[1:]:

        f = h5py.File(fname)

        N = f["Y"].shape[0]
        algo = f.attrs["algo"]
        time = f.attrs["time"]
        error = f.attrs["error"]

        datum = (algo, N, time, error)
        data.append(datum)

    algos = list(set(algo for algo, N, time, error in data))
    algos.sort()

    sizes = list(set(N for algo, N, time, error in data))
    sizes.sort()

    time_curves = {algo: [None] * len(sizes) for algo in algos}
    error_curves = {algo: [None] * len(sizes) for algo in algos}
    for algo, N, time, error in data:

        idx = sizes.index(N)
        time_curves[algo][idx] = time
        if error >= 0:
            error_curves[algo][idx] = error


    print time_curves
    print error_curves

    for algo in algos:
        plt.plot(sizes, time_curves[algo], label=algo)

    ax = plt.axes()
    ax.set_xlabel("Data Size")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime Comparison")

    plt.legend()
    plt.show()


    for algo in algos:
        if algo == "C++ Barnes-Hut":
            continue
        plt.plot(sizes, error_curves[algo], label=algo)

    ax = plt.axes()
    ax.set_xlabel("Data Size")
    ax.set_ylabel("K-L Divergence")
    ax.set_title("Error Comparison")

    plt.legend()
    plt.show()





if __name__ == '__main__':
    main()

