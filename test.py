# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:30:56 2022

@author: yang
"""

import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm
import gensim
from scipy.sparse import csr_matrix, save_npz,load_npz
from data_process import spec
from data_process.spec_to_wordvector import spec_to_wordvector
spectrums=[]
all_files = os.listdir('data/query spectra/')
for i in range(len(all_files)):
    f = 'data/query spectra/' + all_files[i]
    mz=[]
    inten=[]  
    data=pd.read_csv(f,header=None)
 
    for j in range(len(data[0])):
        mz.append(float(round(float(data[0][j]))))
        inten.append(float(data[1][j]))
        M=np.array(mz)
        I=np.array(inten)
        I/= max(I)
                #delete noise
        keep = np.where(I > 0.001)[0]
        M = M[keep]
        I = I[keep]

    if max(M)>1500:
        continue
    else:                
        spectrum = spec.Spectrum(mz=M,intensities=I,
                                metadata={'compound_name': 'substance_measured'+str(all_files[i])})
        spectrums.append(spectrum)
spec.save_as_mgf(spectrums, 'data/meassured_spectra.mgf') 
model_file ="model/references_word2vec.model"
        # Load pretrained model (here dummy model)
model = gensim.models.Word2Vec.load(model_file)
spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)
wordembeddings=[]
spectrums = [s for s in spectrums if s is not None]
for i in tqdm(range(len(spectrums))):
    spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
    embedding=spectovec._calculate_embedding(spectrum_in)
    wordembeddings.append(embedding)
spectra_embeddings=csr_matrix(np.array(wordembeddings))
save_npz('data/meassured_spectra_embeddings.npz',spectra_embeddings)


import time
import hnswlib
import pickle
xq= spectra_embeddings.todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
start_time=time.time()*1000
p = hnswlib.Index(space='l2', dim=dim) 
p.load_index("data/references_index.bin", max_elements =2343378)
end_time=time.time()*1000
print('loadindex_time %.4f'%((end_time-start_time)/100))
start_time=time.time()*1000
# Controlling the recall by setting ef:
p.set_ef(300) # ef should always be > k   ##
# Query dataset, k - number of closest elements (returns 2 numpy arrays)
k=100
I, D = p.knn_query(xq, k)
end_time=time.time()*1000
print('search_time %.4f'%((end_time-start_time)/100))

np.save('data/10compounds_index_results.npy',I)
np.save('data/10compounds_score_results.npy',D)





