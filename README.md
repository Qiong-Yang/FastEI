<div align="center">
<img src="https://github.com/Qiong-Yang/FastEI/blob/main/img/FastEI%20figure.png" width="50%">
</div>


![GitHub](https://img.shields.io/badge/platform-Windows|Linux|MacOS-brightgreen)

![GitHub](https://img.shields.io/badge/license-Apache--2.0 License-orange)

FastEI is an ultra-fast and accurate spectrum matching method, proposed to improve accuracy by Word2vec-based spectrum embedding and boost the speed using hierarchical navigable small world graph (HNSW)

# Motivation

The *in-silico* library with predicted spectra of large-scale molecules can extend the chemical space and increase the coverage immensely when compared with experimental libraries (e.g., NIST 2017 and MassBank libraries). How to rapidly search an *in-silico* library of millions or even tens of millions of spectra while ensuring the accuracy of molecular identification is a new challenge. In this work, a million-molecule scale  *in-silico* library has been builded and an  ultra-fast and accurate search method has been developed (FastEI).

# Depends

Anaconda for python 3.7

conda install gensim

conda install -c rdkit rdkit

conda install -hnswlib

# Usage

**Word2vec model and HNSW index download:**

Please put [HNSW_index.bin](https://zenodo.org/record/6496527/files/2343378_index.bin)  in to **index** directory 

Please put [references_word2vec.model](https://zenodo.org/record/6496527/files/references_word2vec.model)   in to **model** directory 

Take **example.py** as an example for molecular identification.
If you want to identify molecules  based on your spectra, please put your spectra files in to **spectra** directory and run **test.py**.