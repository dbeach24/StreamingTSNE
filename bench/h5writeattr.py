#!/usr/bin/env python

import sys
import h5py


def usage():
    print "usage: %s <fname> key=value [key=value ...]" % sys.argv[0]
    sys.exit(-1)


def main():

    args = sys.argv

    if len(args) < 3:
        usage()

    fname = args[1]

    attrs = {}
    for attr in args[2:]:
        if '=' not in attr:
            usage()
        key, value = attr.split("=")
        attrs[key] = value

    f = h5py.File(fname, "a")

    for key, value in attrs.iteritems():
        f.attrs[key] = value

    f.close()


if __name__ == "__main__":
    main()

