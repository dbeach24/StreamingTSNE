---
title: "Scalable t-SNE: Interim Progress Report #2"
author: "David J. C. Beach"
date: 7/19/2017
geometry: "margin=1in"
#bibliography: references.yaml
reference-section-title: References
link-citations: true
---

# Work Accomplished

The following progress has been made over the previous week:

  * Implementation of Vantage-Point Tree (VP-Tree) data structure in Julia,
    with support for insertion and deletion of single elements.

  * Implementation of Quad-Tree data structure in Julia with insert and delete.
    The center of mass is stored at each level of the tree to support the
    needs of the Barnes-Hut implementation of t-SNE.

  * Started on implementation of Optimization algorithm for streaming version
    of Barnes-Hut algorithm.  Implemented solver for sigma values around each
    point, and equations for $p_{j|i}$ and $q_{j|i}$ from the t-SNE paper.

  * Demonstrated implementation flaw in open-source Julia implementation of dense
    t-SNE algorithm and filed a bug report with the author.  This has
    [received some attention](https://github.com/lejon/TSne.jl/issues/11).

# Plans for Next Week

  * Implement a main loop for the streaming t-SNE implementation.

  * Some demonstration of online optimization with animated graphics.  (May
    only work on small data; will not be optimized.)

# Obstacles / Improvements

  * Creating a working delete mechanism for VP-Trees proved to be challenging, but
    finally have something that makes acceptable performance / completeness
    trade-offs.  When deleting a node, if the node is a leaf, we remove it immediately.
    If the node is an internal node, we mark it as deleted, but do not remove it
    from the tree.  After every insert and delete, any subtree of depth 5 which has
    fewer than 12 nodes is rebuilt, and all marked nodes are collected as garbage.
    This allows the VP-tree to retain some approximate balance while avoiding
    expensive rebuilds of deep portions of the tree.  The cost is that some nodes
    (those which are higher up in the tree) can not be removed.  However, this
    fraction typically comprises only 1-2% of the total nodes in the tree, and
    therefore is an acceptable trade-off.

  * Implementing the complete Barnes-Hut optimizations using the equations
    from the original paper, but in a way that supports streaming is going to
    require some careful thought.
