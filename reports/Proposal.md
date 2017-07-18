---
title: "A Scalable t-SNE Implementation"
subtitle: "Project Proposal"
author: "David J. C. Beach"
date: 6/25/2017
geometry: "margin=1in"
#bibliography: references.yaml
reference-section-title: References
link-citations: true

references:

    - id: tsne
      author:
        - family: "van der Maaten"
          given: Laurens
        - family: Hinton
          given: Geoffrey
      title: Visualizing Data using t-SNE
      issued:
        year: 2008
      type: article-journal
      volume: 9
      container-title: "Journal of Machine Learning Research"

    - id: fast-tsne
      author:
        - family: "van der Maaten"
          given: Laurens
      title: Accelerating t-SNE using Tree-Based Algorithms
      issued:
        year: 2014
      type: article-journal
      volume: 15
      container-title: "Journal of Machine Learning Research"

    - id: atsne
      author:
        - family: Pezzotti
          given: Nicola
        - family: Lelieveldt
          given: Boudewijn P.F.
        - family: "van der Maaten"
          given: Laurens
        - family: Höllt
          given: Thomas
        - family: Eisemann
          given: Elmar
        - family: Vilanova
          given: Anna
      title: "Approximated and User Steerable tSNE for Progressive Visual Analytics"
      issued:
        year: 2017
      type: article-journal
      volume: 23
      issue: 7
      container-title: "IEEE Transactions on Visualization and Computer Graphics"

    - id: dimscale
      author:
        - family: Kruskal
          given: J. B.
      title: "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis"
      issued:
        year: 1964
      volume: 29
      container-title: Psychometrika

    - id: barneshut
      author:
        - family: Barnes
          given: J.
        - family: Hut
          given: P.
      title: "A hierarchical O(N log N) force-calculation algorithm"
      issued:
        year: 1986
      volume: 324
      container-title: Nature

    - id: viszoo
      author:
        - family: Heer
          given: Jeffrey
        - family: Bostok
          given: Michael
        - family: Ogievetsky
          given: Vadim
      title: "A Tour through the Visualization Zoo"
      issued:
        year: 2010
      volume: 8
      issue: 5
      container-title: "ACM Queue"
   
    - id: kldivergence
      author:
        - family: Kullback
          given: S.
        - family: Leibler
          given: R. A.
      title: "On Information and Sufficiency"
      issued:
        year: 1951
      volume: 22
      issue: 1
      container-title: "The Annals of Mathematical Statistics"

    - id: mnist
      author:
        - family: LeCun
          given: Yann
        - family: Cortes
          given: Corinna
        - family: Burges
          given: Christopher J.C.
      title: "The MNIST Database of Handwritten Digits"
      URL: http://yann.lecun.com/exdb/mnist/

    - id: julia
      author:
        - family: Bezanson
          given: Jeff
        - family: Karpinski
          given: Stefan
        - family: Shah
          given: Viral B.
        - family: Edelman
          given: Alan
      title: "Julia: A Fast Dynamic Language for Technical Computing"
      journal: CoRR
      volume: abs/1209.5145
      issued:
        year: 2012
      URL: http://julialang.org/

    - id: vptrees
      author:
        - family: McNutt
          given: Nick          
      title: "`VPTrees.jl`"
      URL: https://github.com/NickMcNutt/VPTrees.jl
      container-title: GitHub

    - id: regiontrees
      author:
        - family: Deits
          given: Robin
      title: "`RegionTrees.jl`"
      URL: https://github.com/rdeits/RegionTrees.jl
      container-title: GitHub

    - id: python-tsne
      title: "t-SNE (Python implementation)"
      author:
        - family: "van der Maaten"
          given: Laurens
      URL: https://lvdmaaten.github.io/tsne/

    - id: julia-tsne
      title: "t-SNE (t-Stochastic Neighbor Embedding)"
      author:
        - family: Jonsson
          given: Leif
      URL: https://github.com/lejon/TSne.jl
      container-title: GitHub

    - id: scikit-tsne
      title: "Scikit-learn (t-SNE implementation)"
      URL: "http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py"

    - id: cppbh
      title: "BHCPP: A C++ Implementation of Barnes-Hut t-SNE"
      URL: "https://github.com/lvdmaaten/bhtsne"


---

# Background

Data sets involving numerous variables are commonplace in scientific
applications and machine learning problems.  The human visual system, however,
is limited to interpreting data in only two or three dimensions, a fact which
has necessitated much work to develop techniques that can embed high-dimensional
data into a low-dimension space.  Visualizations such as
scatter plot matrices and parallel coordinates have proven effective when the
number of dimensions is merely moderate, say 10 or 20 [@viszoo].
However, many emerging problems involve data sets with hundreds or thousands of dimensions.
Early efforts such as multidimensional scaling [@dimscale] have attempted to
solve the problem of high dimensionality by preserving the original distances
in the low-dimension embedding.  Unfortunately, such techniques have suffered
due to their over-emphasis on preserving large distances, often at the expense
of distorting local relationships in the data.

Stochastic Neighbor Embedding (SNE), and especially the more recent Student-T
Stochastic Neighbor Embedding (t-SNE) [@tsne] is an award winning technique
for embedding data of high dimension into a 2-D or 3-D space, while preserving
local relationships in the original data.  This has yielded marked
improvements in the quality of the embedding, especially with regard to its
ability to preserve clusters in the embedded data.

The t-SNE algorithm as original proposed has complexity $\mathcal{O}(n^2)$ [@tsne],
however through the use of approximations and graphical search trees, the
algorithm has been optimized to support processing in $\mathcal{O}(n \log n)$ time
[@fast-tsne].  Very recent efforts have developed an even faster version of the
algorithm, called Accelerated t-SNE (A-tSNE) [@atsne] which enables a trade-off
between runtime performance and accuracy to be made.  These versions
of the t-SNE algorithm have opened up the possibility of visualizing much larger
datasets than was previously practical.

# Goals

The goals of this project are:

  1. to experiment with modifications to the t-SNE algorithm that could allow
     it to be used for online processing with streaming data, and
  
  2. to visually demonstrate what a t-SNE visualization might look like when
     applied to real time streaming data.

The effort will begin with benchmarking existing implementations of t-SNE on a
common data set and recording their relative performance.  Then, a Julia
language [@julia] implementation of the $\mathcal{O}(n \log n)$ accelerated
version of t-SNE [@fast-tsne] will be implemented and compared.  Provided this
effort is successful, further changes will be tested in an attempt to support
visualization of streaming data, using one or more of the following approaches:

  - allow insertion or removal of a single point targeting $\mathcal{O}(\log n)$
    time complexity or better,
  - allow insertion or removal of a small batch of $k$ points targeting
    $\mathcal{O}(k \log n)$ time complexity or better,
  - cluster similar points in the high dimensional space to reduce the number of points
    which must be projected via t-SNE, and
  - employ parallelism which leverages multi-core or cluster hardware for
    improved performance and reduced runtime.

# Testing and Metrics

Implementations will be tested using the MNIST database [@mnist], a set of
scanned handwritten digits encoded as grayscale bitmaps of uniform size.  Testing may
employ a smaller subset of the full MNIST data, depending on practical runtime
constraints, and may also be extended to include other data sets if time
permits.

Results will be plotted, visually inspected for correctness, and benchmarked
for comparative purposes.  In particular, we are interested in measuring the
runtime (wall time) and converged value of the objective function (error) for
each test implementation.

Testing will include three baseline implementations of the original t-SNE algorithm:

 1. the Python implementation provided by the author [@python-tsne],
 2. a Julia translation of this implementation [@julia-tsne], and
 3. the accelerated t-SNE algorithm implemented in Scikit Learn [@scikit-tsne].

The benchmark results of these implementation will be use to compare with the
accelerated Julia implementation to be developed under this proposal.

# Timeline and Deliverables

## Deliverables

This proposal includes the following deliverables:

----------------------------------------------------------------------------------
Date        Milestone                Description
----------  -----------------------  ---------------------------------------------
06-22-2017  Project Intent           Paragraph of intent on discussion board

06-26-2017  Project Proposal         This document

07-10-2017  Interim Progress Report  Progress, Benchmarks, and visuals

07-24-2017  Final Report / Present   TBD
----------------------------------------------------------------------------------

Each of the above deliverables is described briefly:

  1. Project Intent

     The project intent had already been delivered. It consisted of a paragraph
     outlining the idea and a high level on the DS-504 class discussion board, and
     was accepted as a project proposal by Professor Salloum on June 22, 2017.

  2. Project Proposal

     The project proposal (this document) describes the planned project in detail,
     outlining the background, goals, and timeline of the project.

  3. Interim Progress Report

     An interim report will be delivered two weeks after this initial proposal.
     This report will include the results of benchmarking the existing t-SNE
     implementations, as well an update on the progress in developing an
     accelerated version in Julia.

  4. Final Report and Presentation

     The final report for the project will be delivered two weeks after the
     interim report.  It will be accompanied by a final version of the code,
     along with visualizations and timings of test results.  Ideally, a live-
     updating visualization of the streaming algorithm will be presented.

## Task List 

The following outline represents anticipated steps needed to support this
effort, but may be subject to revision as the work progresses.

  * Week 1

    - Develop benchmark script to produce plots, collect runtime and errors
    - Tabulate results
    - Begin work on accelerated t-SNE implementation

  * Week 2

    - Develop accelerated implementation of t-SNE in Julia
    - Interim progress report

  * Week 3

    - Benchmark accelerated t-SNE in Julia
    - Experimental changes to support streaming

  * Week 4

    - Create presentation
    - Write final report

# Overview of t-SNE 

The following section outlines of the original and accelerated versions of the
t-SNE algorithm.  In describing these algorithms we define the following
variables:

  - $n$ points,
  - $d$ dimensions,
  - $X$, a data matrix of size $(n \times d)$, and
  - $Y$, a corresponding matrix of size $(n \times 2)$ representing
    the 2-D embedding of the original input.

We define the conditional probability between two input points $x_i$ and $x_j$
using Gaussian assumptions as follows:

$$ p_{j|i} = {
    \mathrm{exp}(-||x_i - x_j||^2 / 2\sigma^2_i) 
        \over
    \sum_{k \ne i} \mathrm{exp}(-||x_i - x_k||^2 / 2\sigma^2_i)
} $$

Similarly, we define conditional probabilities between the points in the
reduced space.  Because this space has low dimension (typically 2 or 3), the
algorithm employs a Student-T distribution (with 1 degree of freedom) rather
than a Gaussian.  This gives the low-dimension distribution more flexibility
to reposition points which are originally distant, while still encouraging the
preservation of local structure in the high dimensional space.  These
conditional probabilities are as follows:

$$ q_{j|i} = {
    \mathrm{exp}(-||y_i - y_j||^2) 
        \over
    \sum_{k \ne i} \mathrm{exp}(-||y_i - y_k||^2)
} $$

The optimization process seeks to minimize the Kullback-Leibler (K-L)
divergence [@kldivergence] between these two distributions using a gradient
descent technique by updating the position of the $y_i$ points according to
the following formula:

$$ {\partial C \over \partial y_i} = 4 \sum_j {
    (p_{ij} - q_{ij}) (y_i - y_j)
        \over
    1 + ||y_i - y_j||^2
} $$

where $p_{ij}$ and $q_{ij}$ are symmetrized versions of the above conditional
probabilities defined as follows:

  * $p_{ij} = (p_{j|i} + p_{i|j}) / 2$, and
  * $q_{ij} = (q_{j|i} + q_{i|j}) / 2$ .

## Original t-SNE

Here, we outline the t-SNE algorithm as originally proposed in [@tsne]:

### Initialization Phase

  * Use Principal Component Analysis (PCA) to reduce matrix
    $X_\mathrm{orig}$ to 50 dimensions (call this $X$).
  * Compute a standard deviation around each point in $X$.
    - Identify the neighborhood around each row $x_i$ of $X$.
    - This is done by solving for a desired perplexity (default 30) around each point. 
      The solution uses a binary search to solve for the desired perplexity.
  * Precompute pairwise affinities $p_{j|i}$ for each $i,j \in 1 \ldots n$,
    and set $p_{ij} = (p_{j|i} + p_{i|j}) / 2$.
  * Initialize $Y$ with random normal values drawn from $\mathcal{N}(0, 10^{-4} I)$.

### Optimization Phase

  * For $i \in 1 \ldots n$
    - Compute the gradient $\partial C \over \partial y_i$.
    - Compute an updated position $y_i^*$ using the gradient.
  * Update all $y_i \leftarrow y_i^*$.
  * Repeat steps in this phase until some convergence criterion is met.

For brevity, some details of the implementation such as "momentum" have been
omitted from this description.

## Accelerated t-SNE

Using certain observations about the relationships between near and distant
points, and combining these with modern graphical tree-based data structures,
it is possible to create an optimized version of the t-SNE algorithm without
incurring any appreciable loss of quality [@fast-tsne].  In particular, we
observe that:

  1. Very distant points in $X$ have a infinitesimally small joint probability.
     For two such points, $p_{ij} = 0$ provides an an equally good solution.

  2. In $Y$, several distant points $\{y_k\}$ which occur at the same
     approximate angle from some center point $y_i$ have a collective
     contribution to the gradient of $y_i$ which can be effectively summarized
     by a single pseudo-point $y_k^*$ located at their center of mass.

  3. Vantage point trees are a graphical search structure appropriate for
     finding nearby neighbors, and are effective when using high dimensional
     data.  Insertion and proximity searches have time complexity
     $\mathcal{O}(\log n)$.

  4. Quadtrees are a graphical search structure for 2-D data which can be used
     to find nearby points and can also function to summarize points which occur
     at some common angle relative to a center point.  Insertion and range
     searches have time complexity $\mathcal{O}(\log N)$.

Observation (1) is used to accelerate the preprocessing as it avoids an
$\mathcal{O}(n^2)$ loop over the input data, $X$.  Observation (2) is commonly
used to accelerated large $n$-body gravity simulation problems, and is known
as the *Barnes-Hut* approximation algorithm [@barneshut].  Each of these
optimizations are made possible by employing the graphical search structures
structures mentioned (3) and (4).

# Implementation Environment

## Julia Language

Julia [@julia] is a recent language for numerical computing which has been
making inroads in the scientific and open source communities for its high
performance and ease of use.  Although Julia does not yet readily integrate
with massive parallel frameworks such as Apache Spark and Beam, it does
provide parallelism through multiprocessing, and supports processing via
communicating processes on a cluster.  Julia comes with many basic numerical
functions baked-in to the core libraries, with many more functions available
in its packaging system.   In addition, recent versions of Julia have added
support for high-performance features such as syntactic loop fusion and multi-
threading.  This makes Julia a compelling choice for experimentation with
algorithm performance and processing large data sets.

## Tools

The following Julia libraries and projects offer functionality which may
be leveraged by a fast t-SNE implementation in Julia:

* Vantage-Point tree in Julia [@vptrees]
* Quad-tree in Julia [@regiontrees]
* Baseline implementation [@julia-tsne]

