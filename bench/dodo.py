
SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 70000]

def mkdir(name):

    return {
        'name': "mkdir: %s" % name,
        'targets': [name],
        'actions': ["mkdir -p %s" % name],
    }



def task_data():

    yield mkdir("samples")

    for size in SIZES:
        yield {
            'name': "Generate Sample N=%d" % size,
            'file_dep': ["build_mnist.py"],
            'targets': ["samples/mnist_%d.h5" % size],
            'actions': ['python build_mnist.py %d' % size]
        }

