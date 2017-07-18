from doit.task import clean_targets


SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 70000]
MAX_JULIA_N2_SIZE = 8000
MAX_SCIKIT_N2_SIZE = 4000
MAX_SCIKIT_BH_SIZE = 8000
MAX_PYTHON_N2_SIZE = 4000


def mkdir(path):
    return {
        'name': "mkdir: %s" % path,
        'targets': [path],
        'actions': ["mkdir -p %s" % path],
    }


def sample_file(n):
    return "samples/mnist_%d.h5" % n


def build_sample(n):
    return {
        'name': "Generate MNIST Sample N=%d" % n,
        'file_dep': ["bench/build_mnist.py"],
        'targets': [sample_file(n)],
        'actions': ['bench/build_mnist.py %d' % n],
        'clean': [clean_targets],
    }


def build_tsne_julia_n2(n):
    infile = sample_file(n)
    outfile = "outputs/julia_n2_mnist_%d.h5" % n

    return {
        'name': "MNIST (Julia N^2) N=%d" % n,
        'file_dep': ["bench/tsne_julia.jl", infile],
        'targets': [outfile],
        'actions': ["(cd bench; ./tsne_julia.jl ../%s ../%s)" % (infile, outfile)],
        'clean': [clean_targets],
    }


def build_tsne_julia_mod(n):
    infile = sample_file(n)
    outfile = "outputs/julia_mod_mnist_%d.h5" % n

    return {
        'name': "MNIST (Julia Mod) N=%d" % n,
        'file_dep': ["bench/tsne_julia_2.jl", infile],
        'targets': [outfile],
        'actions': ["(cd bench; ./tsne_julia_2.jl ../%s ../%s)" % (infile, outfile)],
        'clean': [clean_targets],
    }


def build_tsne_scikit(n, exact=False):
    method = 'exact' if exact else 'barnes_hut'
    infile = sample_file(n)
    outfile = "outputs/scikit_%s_mnist_%d.h5" % (method, n)

    return {
        'name': "MNIST (Scikit %s) N=%d" % (method, n),
        'file_dep': ["bench/tsne_scikit.py", infile],
        'targets': [outfile],
        'actions': ["bench/tsne_scikit.py %s %s --method=%s" % (infile, outfile, method)],
        'clean': [clean_targets],
    }


def build_tsne_python(n):
    infile = sample_file(n)
    outfile = "outputs/python_n2_mnist_%d.h5" % n

    return {
        'name': "MNIST (Python N^2) N=%d" % n,
        'file_dep': ["bench/tsne_python.py", infile],
        'targets': [outfile],
        'actions': ["bench/tsne_python.py %s %s" % (infile, outfile)],
        'clean': [clean_targets]
    }


def build_tsne_bhtsne(n):
    infile = sample_file(n)
    outfile = "outputs/cpp_bhtsne_mnist_%d.h5" % n

    return {
        'name': "MNIST (C++ BH-TSNE) N=%d" % n,
        'file_dep': ["bench/tsne_bhtsne.py", infile],
        'targets': [outfile],
        'actions': ["bench/tsne_bhtsne.py %s %s" % (infile, outfile)],
        'clean': [clean_targets]
    }


def task_data():

    yield mkdir("samples")
    yield mkdir("outputs")

    for n in SIZES:
        yield build_sample(n)

    for n in SIZES:
        if n <= MAX_JULIA_N2_SIZE:
            yield build_tsne_julia_n2(n)

    for n in SIZES:
        if n <= MAX_JULIA_N2_SIZE:
            yield build_tsne_julia_mod(n)

    for n in SIZES:
        if n <= MAX_SCIKIT_N2_SIZE:
            yield build_tsne_scikit(n, exact=True)

    for n in SIZES:
        if n <= MAX_SCIKIT_BH_SIZE:
            yield build_tsne_scikit(n, exact=False)

    for n in SIZES:
        if n <= MAX_PYTHON_N2_SIZE:
           yield build_tsne_python(n)

    for n in SIZES:
        yield build_tsne_bhtsne(n)

