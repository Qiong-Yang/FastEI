# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:42:57 2022

@author: yang
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 09:05:51 2022

@author: yang
"""
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from data_process import spec
import os
#load predicted EI-MS and save as mgf
spectrums=[]
all_files = os.listdir('/prediected_EIMS_SDF/')
for i in range(len(all_files)):
    f = "/prediected_EIMS_SDF/" + all_files[i]
    suppl = Chem.SDMolSupplier(f)
    for j in tqdm(range(len(suppl))):
        smi = Chem.MolToSmiles(suppl[j])
        try:
            spec_string = suppl[j].GetProp('PREDICTED SPECTRUM')
            A=spec_string.split('\n')
            Y=[x.split(' ') for x in A]
            W=[[float(x[0]),float(x[1])] for x in Y]
            
            c=np.array(W)
            M=np.array([x[0] for x in c])
            I=np.array([x[1] for x in c])
            I/= max(I)
            if max(M)>1500:
                continue
            else:
                
                spectrum = spec.Spectrum(mz=M,intensities=I,
                                metadata={'compound_name': 'substance_chembl'+str(j),'smiles':smi})
                spectrums.append(spectrum)
        except  KeyError:
            continue
spec.save_as_mgf(spectrums, '/predcited_spectrums.mgf')
#load meassured EI-MS and save as mgf
import pandas as pd
all_test_smiles = []
all_test_vectors = []
data=pd.read_csv('/test_11499molecules.csv')
test_SMILES=[]
test_mainSPEC=[]
test_mainpeakindex=[]
test_mainpeakintensity=[]
test_mainspectrum=[]
test_replibSPEC=[]
test_replibpeakindex=[]
test_replibpeakintensity=[]
test_replibspectrum=[]
for i in range(len(data['inchikey'])):
    smi=data['smiles'][i]
    test_SMILES.append(smi)
    #test_mainSPEC.append(data['Spectramain'][i])
    c=data['spectramain'][i]
    d=c.split( )
    intensitymain=[]
    indexmain=[]
    for j in range(len(d)):
        a=d[j].split(':')
        indexmain.append(int(a[0]))
        intensitymain.append(int(a[1]))   
    mzsmain = np.array([float(x) for x in indexmain])
    abundsmain = np.array([float(x) for x in intensitymain])
    abundsmain /= max(abundsmain)   
    spectrummain = spec.Spectrum(mz=mzsmain,intensities=abundsmain,
                                metadata={'compound_name': 'substancemain'+str(i),'smiles':data['smiles'][i]})
    test_mainspectrum.append(spectrummain)
    c=data['spectrareplib'][i]
    d=c.split( )
    intensityreplib=[]
    indexreplib=[]
    for j in range(len(d)):
        a=d[j].split(':')
        indexreplib.append(int(a[0]))
        intensityreplib.append(int(a[1]))
        
    mzsreplib = np.array([float(x) for x in indexreplib])
    abundsreplib = np.array([float(x) for x in intensityreplib])
    abundsreplib /= max(abundsreplib) 
    spectrumreplib = spec.Spectrum(mz=mzsreplib,intensities=abundsreplib,
                                metadata={'compound_name': 'substancereplib'+str(i),'smiles':data['smiles'][i]})
    test_replibspectrum.append(spectrumreplib)
spec.save_as_mgf(test_mainspectrum, '/test11499_mainmeasured_spectrum.mgf')
spec.save_as_mgf(test_replibspectrum, '/test11499_replibmeasured_spectrum.mgf')
