# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:57:57 2021

@author: 15362
"""


import numpy as np
from scipy.sparse import load_npz
import time
import hnswlib
import pickle
xq= load_npz("test_11499_meassured_word_embedings.npz").todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
start_time=time.time()*1000
p = hnswlib.Index(space='l2', dim=dim) 
p.load_index("/2166721_index.bin", max_elements =2166721)
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

np.save('/hnsw_index_results.npy',I)
np.save('/hnsw_score_results.npy',D)

import pandas as pd
test=pd.read_csv('/test_11499.csv')
all_2166721_test_index=test['2166721_index']
s_idx=[]
for i in range(len(I)):
    for j in range(100):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_100: ' +str(len(s_idx)))
s_idx=[]
for i in range(len(I)):
    for j in range(10):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_10: ' +str(len(s_idx)))
s_idx=[]
for i in range(len(I)):
    for j in range(3):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_3: ' +str(len(s_idx)))

s_idx=[]
for i in range(len(I)):
    for j in range(2):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_2: ' +str(len(s_idx)))

s_idx=[]
for i in range(len(I)):
    for j in range(1):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_1: ' +str(len(s_idx)))

#MASS filter added
import numpy as np
ri =list(np.load('/public/home/hpc192301010/08/mass/2166721mass.npy'))
test=pd.read_csv('/public/home/hpc192301010/08/neims_test_11600_main_replib_spec_sdfsmiles_fitered_11499_index_mass.csv')
#I=np.load('D:/gc-ms/08/noweighted_NEIMS_res/3500ri/noweighted_main_3500ri_1500_top_100.npy') 
mass_I=[]
for i in range(11499):
    a=list(I[i])
    ID=[]
    for j in range(len(a)):
        if np.abs(test['extra_mass'][i]-ri[int(a[j])])>=5:
            ID.append(j)
        else:
            pass
    for k in reversed(ID):
        del a[k]
    mass_I.append(a)
from tqdm import tqdm
test['test_3500_result_RIfilter']=''
for i in tqdm(range(len(test['2166721_index']))):
        res=[]
        a=list(test['2166721_index'])
        #a=test['predicted-data-index'][i].strip('[]').split(',')
        #a=[int(x) for x in a]
        for j in range(len(ri_I[i])):
           
            if mass_I[i][j]==a[i]:
                res.append([i,j,a[i]])
            else:
                pass
        test['test_3500_result_RIfilter'][i]=res
c=list(test['test_3500_result_RIfilter'])
d=[]
rank=[]
for i in range(len(c)):
    a=c[i]
    if len(a):
        d.append(i)
        rank.append(a[0][1])
print(len(d))
s=[]
for i in range(len(rank)):
    if rank[i]<100:
        s.append(i)
print('filter_recall_100: ' +str(len(s)))
s=[]
for i in range(len(rank)):
    if rank[i]<10:
        s.append(i)
print('filter_recall_10: ' +str(len(s)))
s=[]
for i in range(len(rank)):
    if rank[i]<3:
        s.append(i)
print('filter_recall_3: ' +str(len(s)))
s=[]
for i in range(len(rank)):
    if rank[i]<2:
        s.append(i)
print('filter_recall_2: ' +str(len(s)))
s=[]
for i in range(len(rank)):
    if rank[i]<1:
        s.append(i)
print('filter_recall_1: ' +str(len(s)))


p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip
### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")

#add expended spectrums and search
xb= load_npz("expanded_176657_predicted_word_sg0.npz").todense().astype('float32')
xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
xb = xb/xb_len
xq= load_npz("conpound10_spectrum.npz").todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
num_elements1 =len(xb)
import time
start_time=time.time()*1000

p = hnswlib.Index(space='l2', dim=dim)  

p.load_index("/2166721_sg0_750.bin", max_elements = 2166721+num_elements1)
end_time=time.time()*1000
print('loadindex_time %.4f'%((end_time-start_time)/100))

import time
start_time=time.time()*1000
print("Adding the news %d elements" % (len(xb)))
p.add_items(xb)
end_time=time.time()*1000
print('add176647_index_time %.4f'%((end_time-start_time)/100))
import time
start_time=time.time()*1000
#p.save_index('/2343381_sg0_750.bin')
end_time=time.time()*1000
print('saveindex_time %.4f'%((end_time-start_time)/100))

import time
start_time=time.time()*1000
p.set_ef(400) 
k=100
# Query dataset, k - number of closest elements (returns 2 numpy arrays)
I, D = p.knn_query(xq, k)
end_time=time.time()*1000
print('search_time %.4f'%((end_time-start_time)/100))
np.save('/hnsw_index_results.npy',I)
np.save('/hnsw_score_results.npy',D)

import pandas as pd
test=pd.read_csv('/measured_compounds10.csv')
all_2166721_test_index=test['2166721_index']
s_idx=[]
for i in range(len(I)):
    for j in range(100):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_100: ' +str(len(s_idx)))
s_idx=[]
for i in range(len(I)):
    for j in range(10):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_10: ' +str(len(s_idx)))
s_idx=[]
for i in range(len(I)):
    for j in range(3):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_3: ' +str(len(s_idx)))

s_idx=[]
for i in range(len(I)):
    for j in range(2):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_2: ' +str(len(s_idx)))

s_idx=[]
for i in range(len(I)):
    for j in range(1):
        if int(I[i][j])==all_2166721_test_index[i]:
            s_idx.append([i,j,int(I[i][j])])
print('recall_1: ' +str(len(s_idx)))

# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
# Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip

### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")



