
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

import os
spectrums=[]
all_files = os.listdir('spectra/')
all_files=sorted(all_files,key = lambda i:int(re.match(r'(\d+)',i).group()))
for i in range(len(all_files)):
    f = 'spectra/' + all_files[i]
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
spec.save_as_mgf(spectrums, 'data/10compounds_meassured_spectra.mgf') 