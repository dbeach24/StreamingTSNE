
SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 70000]


def mkdir(path):
    return {
        'name': "mkdir: %s" % path,
        'targets': [path],
        'actions': ["mkdir -p %s" % path],
    }


def build_sample(n):
    yield {
        'name': "Generate Sample N=%d" % n,
        'file_dep': ["build_mnist.py"],
        'targets': ["samples/mnist_%d.h5" % n],
        'actions': ['python build_mnist.py %d' % n]
    }


def task_data():

    yield mkdir("samples")
    yield mkdir("builds")

    for n in SIZES:
        yield build_sample(n)

