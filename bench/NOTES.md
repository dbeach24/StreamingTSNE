# NOTES


## Scikit Learn t-SNE

Scikit Learn Barnes Hut t-SNE is broken:

https://github.com/scikit-learn/scikit-learn/issues/8582

Also believe results are "wrong" (poor clustering) regardless of whether
you use the "exact" or "barnes_hut" implementations.  Found a related issue:

https://github.com/scikit-learn/scikit-learn/issues/6204



## Barnes-Hut TSNE from author

C++ Implementation on Github with language bindings:

https://github.com/lvdmaaten/bhtsne

Also available as python package: `bhtsne`, however this version was not
compatible with my installed version of `numpy`, so I had to build from
sources.






