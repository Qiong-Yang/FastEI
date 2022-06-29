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
xq= load_npz("/test_11499_meassured_word_embedings.npz").todense().astype('float32')
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

np.save('results/test11499_hnsw_index_results.npy',I)
np.save('results/test_11499hnsw_score_results.npy',D)


p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip
### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")

#add expended spectrums and search
xb= load_npz("data/expanded_176657_predicted_word_embedings.npz").todense().astype('float32')
xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
xb = xb/xb_len
xq= load_npz("data/compounds10_meassured_word_embedding.npz").todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
num_elements1 =len(xb)
import time
start_time=time.time()*1000

p = hnswlib.Index(space='l2', dim=dim)  

p.load_index("index/2166721_index.bin", max_elements = 2166721+num_elements1)
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
p.save_index('index/2343381_index.bin')
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
np.save('results/compounds10_hnswlib_index_results.npy',I)
np.save('results/compounds10_hnswlib_score_results.npy',D)


# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
# Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip

### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")



