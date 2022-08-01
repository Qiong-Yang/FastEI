# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:57:57 2021

@author: yang
"""


import numpy as np
from scipy.sparse import load_npz
import time
import hnswlib
import pickle
xq= load_npz(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/compounds10_measured_word_embeddings.npz')).todense().astype('float32')
xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
xq = xq/xq_len
dim = 500
start_time=time.time()*1000
p = hnswlib.Index(space='l2', dim=dim) 
p.load_index(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+"/data/references_index.bin"), max_elements =2166721)
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

np.save(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/10compounds_index_results.npy'),I)
np.save(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/10compounds_score_results.npy'),D)


p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip
### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")




