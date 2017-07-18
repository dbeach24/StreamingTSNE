---
title: "A Critique: Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces"
author: "David J. C. Beach"
date: 7/12/2017
geometry: "margin=1in"

references:

    - id: vptree
      author:
        - family: Yianilos
          given: Peter N.
      title: Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces
      issued:
        year: 1993
      type: article-journal
      pages: 311-321
      container-title: "Proceedings of the fourth annual ACM-SIAM symposium on Discrete algorithms"

    - id: metrictree
      author:
        - family: Uhlmann
          given: Jeffrey K.
      title: Metric Trees
      issued:
        year: 1991
      container-title: "Applied Mathematics Letters"
      volume: 4
      number: 5
      pages: 61-62

    - id: dynamicvp
      author:
        - family: Fu
          given: Ada W.
        - family: Chan
          given: Polly Mei-shuen
        - family: Cheung
          given: Yin-Ling
        - family: Moon
          given: Yiu Sang
      title: Dynamic vp-Tree indexing for $n$-nearest neighbor search given pair-wise distances
      issued:
        year: 2000
      pages: 154-173
      container-title: "The VLDB Journal"
      volume: 9

    - id: searchingmetric
      author:
        - family: Chávez
          given: Edgar
        - family: Navarro
          given: Gonzalo
        - family: Baeza-Yates
          given: Ricardo
        - family: Marroquín
          given: José Luis
      title: Searching in Metric Spaces
      container-title: ACM Computing Surveys
      volume: 33
      issued:
        year: 2001
      pages: 273-321      

    - id: vpworkloads
      author:
        - family: Mao
          given: Rui
        - family: Lei
          given: Ving I.
        - family: Ramakrishnan
          given: Smriti
        - family: Xu
          given: Weijia
        - family: Miranker
          given: Daniel P.
      title: On Metric-Space Indexing and Real Workloads
      year: 2005

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


#bibliography: references.yaml
reference-section-title: References
link-citations: true
---


# Overview

Yianilos introduces a data structure and associated algorithms for indexing
and seaching neighborhoods in a metric space with $\mathcal{O}(N \log N)$
complexity in time, and $\mathcal{O}(N)$ requirements in space [@vptree].  The
data structure is referred to as a Vantage-Point Tree (or VP Tree).  His work
is frequently cited in literature relating to nearest neighbor searches, and
would likely be considered by many to be an early, seminal work on the topic.
Although his work is highly praiseworthy, it may not be entirely novel, and
Yianilos admits as much: the idea was simultaneously discovered by Uhlmann
[@metrictree], who referred to a nearly identical structure as a *metric*
tree.

The problem of efficiently indexing and searching metric spaces addresses an
important need, and is widely applicable to many areas such as statistical
learning, computer vision, and document retrieval.  The structure and
algorithms presented are general enough to work with data in any metric space.
High dimensional Euclidean spaces and non-Euclidean metric spaces can be
efficiently indexed and searched using the techniques described in this paper.

There are numerous uses for such a solution, particularly that of making
k-nearest neighbor or fixed radius searches efficient.  The goal of our final
project is to implement an efficient online version of the t-SNE algorithm for
dimension reduction [@tsne].  To be efficient, this algorithm must have a
spatial index of the input data (typically of high dimension), and this search
must be able to quickly identify other points in a local neighborhood.
Vantage-Point trees address this need, and are indeed used in as part of the
Barnes-Hut accelerated version of the algorithm [@fast-tsne].

# Assumptions

Vantage-Point trees require that the data lie in a metric space.
Specifically, it must be possible to measure distances between any two points
using a metric function which adheres to typical notions of a distance
function (non-negative, commutative) and require that is adheres to the
triangle inequality.  Formally, for any points $a, b, c$ in a metric space,
the metric function $d(x, y)$ must adhere to these properties:

  1. $d(a,b) \ge 0$,
  2. $d(a,a) = 0$,
  3. $d(a,b) = d(b,a)$, and
  4. $d(a,c) <= d(a,b) + d(b,c)$ (i.e. the triangle inequality).

Norms $L_1$, $L_2$, $L_\infty$, etc. in vector spaces meet this requirement, as well as
several non-Euclidean distance metrics such as cosine distance (a
*pseudometric*), Hamming distance (for strings), tree distance, or other
shortest path graph distances.  Hence, the fact that this search structure
applies to general metric spaces makes it applicable to a wide range of
problems.

A shortcoming of the algorithm as presented is that it assumes a static
database of points which must be fully known prior to indexing.  This
assumption is incompatible with streaming data problems.  Yianilos does not
present any algorithms for inserting, or removing points, or merging or
re\-balancing vantage point trees.  Other work has proposed algorithms for
doing this [@dynamicvp], however it has been suggested that the problem of
efficiently merging vantage point trees, in particular, is difficult and
remains open.

# Evaluation

Early in the paper, Yianilos graphically demonstrates the different effect
that is obtained when hierarchically partitioning a 2-D space with randomly
selected points using KD-trees versus VP\-trees.  This illustration presents a
contrast between the axis-aligned boundaries employed by KD-trees and the
arc\-shaped boundaries created by VP\-trees.  This illustration also shows
Yianilos's preference for selecting vantage points in "corners" of the space.
He later expounds on the justifications for this selection criteria.  These
arguments hold up well if the data are uniformly distributed in a low
dimension.  However, it is not clear that is is reasonable to assume that most
high-dimensional data are approximately uniformly distributed, or that
selecting a vantage point near the "corner" actually produces the best results
for many real-world data sets.  Others have suggested that selecting a more
central vantage point may produce better results, particularly in high
dimensional settings, or when the data exhibit clustering patterns
[@vpworkloads].

In addition, Yianilos offers two evaluations of Vantage\-Point trees to
illustrate their utility in indexing a metric space. He first applies the
structure to an image retrieval problem, selecting images from a database
which are similar to a given query image.  Second, he claims that the
algorithm has been tested using cosine similarity and using a Euclidean
distance that has been normalized by $||X|| + ||Y||$.  These later test
metrics use random data points, and no concrete results are presented by the
author, other than to say the performance "corresponds well qualitatively to a
standard Euclidean setting." This later evaluation especially, is devoid of
any details, and could be considered a shortcoming of the paper.  There are no
analyses which test the algorithm using metrics that derive from non-Euclidean
data, such as Hamming distance or graph distances.  The inclusion of such
examples would have made for a more comprehensive evaluation.

When Yianilos introduces the Vantage\-Point tree concept, he begins with the
procedure for constructing a simple Vantage\-Point tree, a vp\-tree.  He then
moves on to define two more sophisticated tree structures, the
$\mathrm{vp^{s}}$\-tree which keeps a history of distances from sub\-nodes to
parent nodes going all the way back to the root, and the
$\mathrm{vp^{sb}}$\-tree which is an extension of the $\mathrm{vp^{s}}$-tree
which incorporates buckets at the leaf levels of the tree.  Only then does the
author see fit to describe the search algorithm, which he claims is for the
simple vp-tree.  However, the search algorithm as written depends on bounds
attributes being stored at each node, which the author simply explains as the
case where "a node contains subspace bounds for each child."  Confusingly,
none of the aforementioned tree building algorithms actually describe the
computation of these bounds, and so the task of determining the exact
algorithm which constructs the tree required by the search algorithm is left
in part as an exercise to the reader (or implementer).  The author could have
taken more care to complete his algorithms, use more consistent notation, and
present the algorithms in a more natural progression.

# Related Work

Since the publication of this paper, much work has been done to develop
algorithms to support exact and approximate neighborhood searches in both
Euclidean spaces and general metric spaces [@searchingmetric].  Among these
related developments include the prior development of BKT trees, the Multi
Vantage Point (MVP) Tree, the excluded middle vantage point forest (VPF),
the Generalized Hyperplane Tree (GHT), and the $n$-ary GHT (GNAT).  The
preceding list of names and acronyms should be taken as a meager
indication of the vast research effort that has gone into indexing and
searching metric spaces in recent decades.


