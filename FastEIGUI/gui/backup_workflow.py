# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:46:14 2022

@author: yang
"""

#先加载库里面的谱转化后的1*500的向量一级相应的SMILES
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz,load_npz

vectors= np.load('data/all_2343378_binning_vectors.npy').astype('float32')
with open('data/spectrumdb.dat', 'wb') as outfile:
    pickle.dump(csr_matrix(vectors), outfile, pickle.HIGHEST_PROTOCOL)

smiles1=list(np.load('data/all_2343378_smiles.npy', allow_pickle=True))
comID = ['Comp_{}'.format(i) for i in range(len(smiles1))]
compounds = pd.DataFrame({'CompID': comID, 'SMILES': smiles1})
compounds.to_csv('data/compounds.csv', index = False)


#将实验谱图通过已经训练好的word2vec模型转化为x向量
import os
from data_process import spec
from tqdm import tqdm
import re

spectrums_m=[]
for i in tqdm(range(len(smiles1))):
    v = vectors[i,:]
    k = np.where(v > 0)[0]
    mz = k.astype(float)
    inten = v[k].astype(float)
    spectrum = spec.Spectrum(mz=mz, intensities=inten)
    spectrums_m.append(spectrum)

from data_process.spec_to_wordvector import spec_to_wordvector
import gensim
model_file = "data/word2vecsg0.model"

# Load pretrained model (here dummy model)
model = gensim.models.Word2Vec.load(model_file)
spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)
word2vectors=[]
spectrums = spectrums_m
for i in tqdm(range(len(spectrums))):
    spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
    vetors = spectovec._calculate_embedding(spectrum_in)
    word2vectors.append(vetors)
word2vectors = np.array(word2vectors)

#利用谱转化后的向量建库
import hnswlib
dim = 500
xb = word2vectors
num_elements = len(xb)
ids = np.arange(num_elements)
# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip
# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 600, M = 64)

import time
start_time=time.time()*1000
# Element insertion (can be called several times):
p.add_items(xb, ids)
p.set_ef(200) # ef should always be > k   ##
end_time=time.time()*1000
print('add_index_time %.4f'%((end_time-start_time)/100))

p.save_index('data/database.bin')
