
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:05:51 2022

@author: yang
"""
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import sys
sys.path.append("..")
from data_process import spec
import sqlite3
import os
gradedb = sqlite3.connect(os.path.abspath(os.path.join(os.getcwd(), ".."))+"/data/IN_SILICO_LIBRARY.db")
cursor=gradedb.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
Tables=cursor.fetchall()
print(Tables)
content = cursor.execute("SELECT COMPID, SMILES, MZS,INTENSITYS from IN_SILICO_LIBRARY")
#load predicted EI-MS and save as mgf
spectrums=[]
for row in tqdm(cursor):
    smi =  row[1]
    X=row[2][1:-1].split(', ')
    M=np.array([float(x) for x in X])
    Y=row[3][1:-1].split(', ')
    I=np.array([float(y) for y in Y])
    I/= max(I)
    if max(M)>1500:
        continue
    else:
                
        spectrum = spec.Spectrum(mz=M,intensities=I,
                                metadata={'compound_id': row[0],'smiles':smi})
        spectrums.append(spectrum)
   
spec.save_as_mgf(spectrums, os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/predcited_spectrums.mgf'))
cursor.close()

import os
import pandas as pd
spectrums=[]
all_files = os.listdir(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/query spectra_csv/'))
for i in range(len(all_files)):
    f = os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/query spectra_csv/') + all_files[i]
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
spec.save_as_mgf(spectrums,  os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/10compounds_measured_spectra.mgf') )
