# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:56:00 2022

@author: yang
"""


import pandas as pd
import os
spectrums_m=[]
all_files = os.listdir('data/query_spectra_txt/')
#all_files=sorted(all_files,key = lambda i:int(re.match(r'(\d+)',i).group()))
smiles=''
for i in range(len(all_files)):
    f = 'data/query_spectra_txt/' + all_files[i]
    mz=[]
    inten=[]  
    with open(f, "r") as f:
        data = []   
        for line in f:        
            data.append(line.rstrip())  
        
    #index=[]
        for j in range(len(data)):
            if data[j]=='m/z	Absolute Intensity	Relative Intensity':
                index=j
            else:
                pass
            #num=data[i][10:]
        for k in range(index+1,len(data)):
            mz.append(float(round(float(data[k].split('\t')[0]))))
            inten.append(float(data[k].split('\t')[2]))
        df={"m/z" : mz,"Relative Intensity" :inten}
        df=pd.DataFrame(df)
        df.to_csv('data/query spectra_csv/'+str(all_files[i])[0:1]+'.csv',index=False,header=False)