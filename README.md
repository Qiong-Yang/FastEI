

<div align="center">
<img src="https://github.com/Qiong-Yang/FastEI/blob/main/img/FastEI%20figure.png" width="50%">
</div>


![GitHub](https://img.shields.io/badge/license-Apache--2.0%20License-orange)

FastEI is an ultra-fast and accurate spectrum matching method, proposed to improve accuracy by Word2vec-based spectrum embedding and boost the speed using hierarchical navigable small world graph (HNSW)

# Motivation

The *in-silico* library with predicted spectra of large-scale molecules can extend the chemical space and increase the coverage immensely when compared with experimental libraries (e.g., NIST 2017 and MassBank libraries). How to rapidly search an *in-silico* library of millions or even tens of millions of spectra while ensuring the accuracy of molecular identification is a new challenge. In this work, a million-molecule scale  *in-silico* library has been builded and an  ultra-fast and accurate search method has been developed (FastEI).

# Depends


[Anaconda for Python 3.7](https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86_64.exe)

conda install -c conda-forge gensim

conda install -c rdkit rdkit

conda install -c conda-forge hnswlib


# Installation

The current install version of FastEI only supports Windows 64-bit version. It has been test on windows 7, windows 10.

Install software: [FastEI-GUI-1.0.0-Windows.exe](https://github.com/Qiong-Yang/FastEI/releases/tag/v1.0.0-beta)

#### Note: All the database, HNSW index, Word2vec model and query spectra examples are included in the installation package. So there are four files needed to be downloaded, including FastEI-GUI.exe, FastEI-GUI-1.bin,  FastEI-GUI-2.bin, and FastEI-GUI-3.bin. It will take some time to install. Please be patient.

# Development version

1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)   

2. Install [Git](https://git-scm.com/downloads)  

3. Open commond line, clone the repository and enter:  

       git clone https://github.com/Qiong-Yang/FastEI.git
       cd FastEI

4. Create environment and install dependency with the following commands :  

      conda env create -f FastEI.yml 
      conda activate FastEI

5. Run FastEI.py:  
        

       cd GUI/ui
       python FastEI.py


# Usage

## Software



The video for using the FastEI is available at the [video folder](https://github.com/Qiong-Yang/FastEI/blob/main/video).

https://user-images.githubusercontent.com/46306770/200098429-dd42b2ba-db06-4f8f-b16f-bfbabb4b17e4.mov


For the details on how to use FastEI, please check [Ducomentation](https://github.com/Qiong-Yang/FastEI/blob/main/Documentation.pdf). 

## Code

**Database, Word2vec model and HNSW index download:**

Please put [IN_SILICO_LIBRARY.db](https://zenodo.org/record/6778379/files/IN_SILICO_LIBRARY.db)  , [references_index.bin](https://zenodo.org/record/6778379/files/references_index.bin)  and [references_word2vec.model](https://zenodo.org/record/6778379/files/references_word2vec.model)   into **data**  directory.

Take **example.py** （**example.ipynb**）as an example for molecular identification.
If you want to identify molecules  based on your spectra, please put your spectra files in to **spectra** directory and run **test.py**.

# Contact

Yang qiong   

E-mail: 192301010@csu.edu.cn

