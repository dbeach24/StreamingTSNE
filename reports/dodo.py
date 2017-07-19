from doit.task import clean_targets


def mkdir(path):
    return {
        'name': "mkdir: %s" % path,
        'targets': [path],
        'actions': ["mkdir -p %s" % path],
    }



def make_pdf(mdfile, pdf, toc=False):

    args = [
        "-S",
    ]

    if toc:
        args.append("--toc")
        args.append("--tod-depth 2")

    args += [
        "--mathjax",
        "--filter pandoc-citeproc",
        "--filter pandoc-eqnos",
        "--csl citation-style.csl",
        "--highlight-style=tango"
    ]

    return {
        'name': "Build PDF: %s" % pdf,
        'file_dep': [mdfile],
        'targets': [pdf],
        'actions': ["pandoc -f markdown %s -o %s %s" % (mdfile, pdf, " ".join(args))],
    }


def task_data():

    yield mkdir("pdf")

    yield make_pdf("Proposal.md", "pdf/Proposal.pdf")
    yield make_pdf("Progress1.md", "pdf/Progress1.pdf")
    yield make_pdf("Critique.md", "pdf/Critique.pdf", toc=False)
    yield make_pdf("Progress2.md", "pdf/Progress2.pdf")
