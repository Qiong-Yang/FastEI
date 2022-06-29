# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:18:26 2022

@author: yang
"""

import numpy as np
from tqdm import tqdm
from data_process import spec
from data_process.spec_to_wordvector import spec_to_wordvector
import os
import gensim
spectrums=list(spec.load_from_mgf("data/predcited_spectrums.mgf"))
import pandas as pd
print(len(spectrums))
model_file ="model/references_word2vec.model"
        # Load pretrained model (here dummy model)
model = gensim.models.Word2Vec.load(model_file)
spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)

word2vectors=[]
word_smiles=[]
spectrums = [s for s in spectrums if s is not None]
for i in tqdm(range(len(spectrums))):
    spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
    vetors=spectovec._calculate_embedding(spectrum_in)
    word_smiles.append(spectrum_in.metadata['smiles'])
    word2vectors.append(vetors)
np.save("data/all_smiles.npy", word_smiles)
from scipy.sparse import csr_matrix, save_npz
word_vec=csr_matrix(np.array(word2vectors))
save_npz('data/all_predicted_word_embeddings.npz', word_vec)

spectrums=list( spec.load_from_mgf('data/10compounds_meassured_spectra.mgf'))
print(len(spectrums))
word2vectors=[]
spectrums = [s for s in spectrums if s is not None]
for i in tqdm(range(len(spectrums))):
    spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
    vetors=spectovec._calculate_embedding(spectrum_in)
    word2vectors.append(vetors)
np.save('data/compounds10_meassured_smiles.npy', word_smiles)
from scipy.sparse import csr_matrix, save_npz
word_vec=csr_matrix(np.array(word2vectors))
save_npz('data/compounds10_meassured_word_embeddings.npz', word_vec)
