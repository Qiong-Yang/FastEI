# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:18:26 2022

@author: yang
"""

import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from data_process import spec
import os
import gensim
spectrums=list(spec.load_from_mgf(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+"/data/predcited_spectrums.mgf")))
reference_documents = [spec.SpectrumDocument(s, n_decimals=0) for s in spectrums]
from data_process.model_building import train_new_word2vec_model    
model_file =os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+ "/model/references_word2vec.model")
model = train_new_word2vec_model(reference_documents, iterations=[40,60,80], filename=model_file,vector_size=500,
                                 workers=10, progress_logger=True)
 


 

