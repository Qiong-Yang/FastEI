# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:30:56 2022

@author: yang
"""

import pandas as pd
import os
import numpy as np
from data_process import spec
import re
from tqdm import tqdm
from data_process.spec_to_wordvector import spec_to_wordvector
import os
import gensim
from scipy.sparse import csr_matrix, save_npz,load_npz
spectrums=[]
all_files = os.listdir('/test/spctrums_txt/')
all_files=sorted(all_files,key = lambda i:int(re.match(r'(\d+)',i).group()))
for i in range(len(all_files)):
    f = '/test/spctrums_txt/' + all_files[i]
    mz=[]
    inten=[]  
    with open(f, "r") as f:
        data = []   
        for line in f:        
            data.append(line.rstrip())  
 
    for j in range(len(data)):
        if data[j]=='m/z	Absolute Intensity	Relative Intensity':
            index=j
            for k in range(index+1,len(data)):
                mz.append(float(round(float(data[k].split('\t')[0]))))
                inten.append(float(data[k].split('\t')[2]))
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
spec.save_as_mgf(spectrums, '/data/compounds10meassured_spectrum.mgf') 
model_file ="/model/word2vec.model"
        # Load pretrained model (here dummy model)
model = gensim.models.Word2Vec.load(model_file)
spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)
word2vectors=[]
spectrums = [s for s in spectrums if s is not None]
for i in tqdm(range(len(spectrums))):
    spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
    embedding=spectovec._calculate_embedding(spectrum_in)
    word2vectors.append(embedding)
word_vec=csr_matrix(np.array(word2vectors))
save_npz('/data/compounds10_meassured_word_embedding.npz', word_vec)


import time
import hnswlib
import pickle
xq= load_npz('data/compounds10_meassured_word_embedding.npz').todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
start_time=time.time()*1000
p = hnswlib.Index(space='l2', dim=dim) 
p.load_index("/index/2343378_index.bin", max_elements =2166721)
end_time=time.time()*1000
print('loadindex_time %.4f'%((end_time-start_time)/100))
import time
start_time=time.time()*1000
# Controlling the recall by setting ef:
p.set_ef(300) # ef should always be > k   ##
# Query dataset, k - number of closest elements (returns 2 numpy arrays)
k=100
I, D = p.knn_query(xq, k)
end_time=time.time()*1000
print('search_time %.4f'%((end_time-start_time)/100))

np.save('results/hnsw_index_results.npy',I)
np.save('results/hnsw_score_results.npy',D)

       