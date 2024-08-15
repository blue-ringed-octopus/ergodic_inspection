# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:34:26 2024

@author: hibad
"""

import pickle 
import matplotlib.pyplot as plt
with open('detections.pickle', 'rb') as f:
    dat = pickle.load(f)

region_index = dat["region"]        
p_region = []

for p in dat["p"]:
    p_region.append([max(p[idx]) for idx in region_index.values()])
    
    