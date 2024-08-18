# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:34:26 2024

@author: hibad
"""

import pickle 
import matplotlib.pyplot as plt
import numpy as np

with open('detections.pickle', 'rb') as f:
    dat = pickle.load(f)

region_index = dat["region"]        
p_region = []

for p in dat["p"]:
    p_region.append([np.quantile(p[idx],1) for idx in region_index.values()])
    
p_region=np.array(p_region)    

for i in range(len(region_index)):
    plt.plot(p_region[:,i], label=str(i))
    plt.legend()