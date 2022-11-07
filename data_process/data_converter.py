import numpy as np
import pandas as pd

def read_jdx_to_csv(file, save_path):
    with open(file) as jdx:
        lines = jdx.readlines()
    spectrum_list = []
    mz, inten = [], []
    for l in lines:
        l = l.replace('\n', '')
    
        if l.startswith('##PAGE'):
            page = l.split(' ')[-1].replace('=','_').replace('.', '_')
        
        if not l.startswith('##'):
            data = [k for k in l.replace(' ', '').split(',') if k != '']
            if len(data) % 2 == 1:
                raise ValueError('the peaks are not paired')
            else:
                n = int(len(data) / 2)
                mz += [float(data[k]) for k in np.arange(n) * 2]
                inten += [float(data[k]) for k in np.arange(n) * 2 + 1]

        if l.startswith('##') and len(mz) > 0:
            df = pd.DataFrame({"m/z" : np.array(mz), "Relative Intensity" : np.array(inten) / np.max(inten)})
            df.to_csv(save_path + '/{}.csv'.format(page), index = False, header = False)
            mz, inten = [], []
        

def read_msp_to_csv(file, save_path):
    with open(file) as msp:
        lines = msp.readlines()

    mz, inten = [], []
    for l in lines:
        l = l.replace('\n', '')
        l = l.lower()
    
        if l == '':
            continue
    
        if l.startswith('name:'):
            mz, inten = [], []
            name = l.split(': ')[-1]
        
        if l.startswith('num peaks:'):
            n_peaks = int(l.split(': ')[-1])
        
        if ':' not in l:
            mz.append(float(l.split(' ')[0]))
            inten.append(float(l.split(' ')[1]))
    
        if len(mz) > 0:
            if len(mz) == n_peaks:
                df = pd.DataFrame({"m/z" : np.array(mz), "Relative Intensity" : np.array(inten) / np.max(inten)})
                df.to_csv(save_path + '/{}.csv'.format(name), index = False, header = False)


def read_txt_to_csv(file, save_path):
    with open(file, "r") as txt:
        data = [] 
        mz=[]
        inten=[]
        name='example'
        for line in txt:        
            data.append(line.rstrip())
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
        df.to_csv(save_path + '/{}.csv'.format(name), index = False, header = False)

