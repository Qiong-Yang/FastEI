#   FastEI
An ultra-fast and accurate spectrum matching (FastEI) method was proposed to improve accuracy by Word2vec-based spectrum embedding and boost the speed using hierarchical navigable small world graph (HNSW)
# Motivation
Spectrum matching is widely used for mass spectrometry-based compound identification. However, this approach fails to identify molecules that are not in the existing libraries. Only approximate 20% of chromatographic peaks extracted from GC-MS dataset can be identified by spectrum matching against libraries based on similarity metrics1. The bottleneck lies on the compound coverage of spectral libraries.One solution is to generate in-silico spectra to extend spectral libraries.The in-silico library with predicted spectra of large-scale molecules can extend the chemical space and increase the coverage immensely when compared with experimental libraries (e.g., NIST 2017 and MassBank libraries). It leads to another challenge: how to rapidly search an in-silico library of millions or even tens of millions of spectra, also ensure the accuracy of the molecular identification.Therefore, a million-molecule scale  in-silico library has been builded and a fast and accurate search method has been developed (FastEI).
# Depends
Anaconda for python 3.7

conda install gensim

conda install -c rdkit rdkit

conda install -hnswlib
# Usage

test