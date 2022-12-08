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
xb= load_npz(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/all_predicted_word_embeddings.npz')).todense().astype('float32')
xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
xb = xb/xb_len
dim = 500
num_elements =len(xb)
ids = np.arange(num_elements)
# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip
# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 800, M = 64)
import time
start_time=time.time()*1000
# Element insertion 
p.add_items(xb, ids)
# Controlling the recall by setting ef:
p.set_ef(300) # ef should always be > k   ##
end_time=time.time()*1000
print('add_time %.4f'%((end_time-start_time)/100))


start_time=time.time()*1000
p.save_index(os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".."))+'/data/references_index.bin'))
end_time=time.time()*1000
print('saveindex_time %.4f'%((end_time-start_time)/100))
# Index objects support pickling
p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip
### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")