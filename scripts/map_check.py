# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:31:35 2024

@author: hibad
"""

import yaml 
import numpy as np 
from Lie import SE3 
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d 

def read_prior(features):  
    prior = {}         
    z = np.zeros(6*len(features))
    idx_map = {}
    children = {}
    for i, (feature_id, pose) in enumerate(features.items()):
        idx_map[feature_id] = i
        M = np.eye(4)
        M[0:3,0:3] = Rot.from_euler('xyz', pose["orientation"]).as_matrix()
        M[0:3,3] =  pose["position"]
        z[6*i:6*i+6]=SE3.Log(M)
        children[feature_id] = M
        
    prior["z"] = z
    prior["idx_map"] = idx_map
    prior["children"] = children
    prior["cov"] = np.eye(len(z))* 0.0001
    return prior

with open('../resources/tag_loc_physical.yaml', 'r') as file:
    features = yaml.unsafe_load(file)
    
prior = read_prior(features)
