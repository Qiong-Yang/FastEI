
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:05:51 2022

@author: yang
"""
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from data_process import spec
import sqlite3
gradedb = sqlite3.connect("data/IN_SILICO_LIBRARY.db")
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
   
spec.save_as_mgf(spectrums, 'data/predcited_spectrums.mgf')
cursor.close()

