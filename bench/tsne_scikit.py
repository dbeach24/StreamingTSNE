# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

from time import time
import h5py
import numpy as np

from sklearn import manifold, datasets

def main(in_name, out_name):

    f = h5py.File(in_name, "r")
    X = f["X"].value.T
    n_samples = X.shape[1]
    labels = f["labels"].value

    n_neighbors = 20
    n_components = 2

    start = time()
    tsne = manifold.TSNE(
        n_components=n_components,
        perplexity=n_neighbors,
        init="pca",
        random_state=0
    )
    Y = tsne.fit_transform(X)
    stop = time()

    error = tsne.kl_divergence_

    print("t-SNE: %.2g sec" % (stop - start))
    print("error: %.2g" % error)

    outf = h5py.File(out_name, "w")
    outf["Y"] = Y
    outf["labels"] = labels
    outf.attrs["algo"] = "Scikit Barnes-Hut"
    outf.attrs["input"] = in_name
    outf.attrs["time"] = (stop - start)
    outf.attrs["error"] = error


if __name__ == "__main__":
    input_name = "mnist_2500.h5"
    output_name = input_name[:-3] + ".scikit.h5"
    main(input_name, output_name)


#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import NullFormatter

# Next line to silence pyflakes. This import is needed.
#Axes3D


# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

# plt.show()