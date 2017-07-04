#!/usr/bin/env python

import sys
import struct
import random
import gzip

import h5py

import numpy as np


def parse_images(f):
    """
    IMAGE DATA
    ----------
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000803(2051) magic number 
    0004     32 bit integer  60000            number of images 
    0008     32 bit integer  28               number of rows 
    0012     32 bit integer  28               number of columns 
    0016     unsigned byte   ??               pixel 
    0017     unsigned byte   ??               pixel 
    ........ 
    xxxx     unsigned byte   ??               pixel
    """

    # read first 4 integer fields (16 bytes) from header
    head = f.read(16)

    magic, nimages, nrows, ncols = struct.unpack(">iiii", head)
    print magic, nimages, nrows, ncols

    imagesize = nrows * ncols

    result = np.zeros((nimages, imagesize), dtype=np.uint8)

    for i in xrange(nimages):
        data = f.read(imagesize)
        image = np.fromstring(data, dtype=np.uint8)
        result[i] = image

    return result


def parse_labels(f):
    """
    LABEL DATA
    ----------
    [offset] [type]          [value]          [description] 
    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    0004     32 bit integer  60000            number of items 
    0008     unsigned byte   ??               label 
    0009     unsigned byte   ??               label 
    ........ 
    xxxx     unsigned byte   ??               label
    """

    # read first 2 integer fields (8 bytes) from header
    head = f.read(8)
    magic, nlabels = struct.unpack(">ii", head)
    print magic, nlabels
    data = f.read(nlabels)
    return np.fromstring(data, dtype=np.uint8)



def main(samp_size):
    train_images = parse_images(gzip.GzipFile("mnist_data/train-images-idx3-ubyte.gz"))
    test_images = parse_images(gzip.GzipFile("mnist_data/t10k-images-idx3-ubyte.gz"))
    all_images = np.vstack([train_images, test_images])
    print all_images.shape

    train_labels = parse_labels(gzip.GzipFile("mnist_data/train-labels-idx1-ubyte.gz"))
    test_labels = parse_labels(gzip.GzipFile("mnist_data/t10k-labels-idx1-ubyte.gz"))
    all_labels = np.hstack([train_labels, test_labels])
    print all_labels.shape

    N = all_images.shape[0]
    assert N == all_labels.shape[0]

    random.seed(42)
    samp = random.sample(range(N), samp_size)

    samp_images = all_images[samp]
    samp_labels = all_labels[samp]

    outf = h5py.File("samples/mnist_%d.h5" % samp_size, "w")
    outf["X"] = samp_images
    outf["labels"] = samp_labels
    outf.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
