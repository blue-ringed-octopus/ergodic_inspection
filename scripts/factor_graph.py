# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:39:29 2024

@author: hibad
"""
import numpy as np
from copy import deepcopy
from numpy.linalg import inv

class Node:
    def __init__(self, node_id, M, node_type):
        self.type=node_type
        self.M=M
        self.cov=np.zeros((6,6))
        self.id=node_id
        self.local_map=None
        self.pruned=False 
        self.factor={}
       
    def copy(self):
        return deepcopy(self)
     
class Factor:
    def __init__(self, id_, parent_id, children_ids, feature_ids, z, sigma, idx_map ):
        self.prior = parent_id == None
        self.parent_id = parent_id
        self.children_ids = children_ids
        self.feature_ids = feature_ids
        self.z = z
        self.omega = inv(sigma)
        self.omega = (self.omega.T+self.omega)/2
        self.pruned = False
        self.n_features = len(feature_ids)
        self.n_poses =  len(children_ids)
        self.idx_map = idx_map
        self.id = id_
        
    def copy(self):
        return deepcopy(self)
    
    def get_jacobian(self, M_init):
        pass 
            
class Factor_Graph:                
    def __init__(self, horizon, forgetting_factor):
        self.prior_factor = Factor(0, [],[], [], None, np.eye(2),  {"features":{}, "pose":{}})
        self.pose_nodes={}
        self.key_pose_nodes={}
        self.factors={}
        self.feature_nodes={}
        self.horizon = horizon
        self.current_pose_id = -1
        self.current_factor_id = 1
        self.forgetting_factor = forgetting_factor
    
    def add_node(self, M, node_type, feature_id=None, key_node = False):
        i=self.current_pose_id+1
        if node_type=="pose":
            node=Node(i,M, node_type)
            self.pose_nodes[i]=node
            if key_node:
                self.key_pose_nodes[i]=node
            self.current_pose_id = i
                
        if node_type=="feature":
            node=Node(feature_id,M, node_type)
            self.feature_nodes[feature_id]=node
        return self.current_pose_id
    
    def add_prior_factor(self, z, sigma, pose_idx_map ,feature_idx_map):
        idx_map={"pose": pose_idx_map, "features": feature_idx_map}
        # children = [self.pose_nodes[id_] for id_ in pose_idx_map.keys()]
        # features = [self.feature_nodes[id_] for id_ in feature_idx_map.keys()]
                  
        factor = Factor(self.prior_factor.id, None, list(pose_idx_map.keys()),list(feature_idx_map.keys()) ,z,sigma, idx_map)  
            
        self.prior_factor = factor
                    
    def add_factor(self, parent_id, child_id, feature_ids, z, sigma, feature_idx_map):
        idx_map={"pose": {child_id: 0}, "features": feature_idx_map}
            
        factor = Factor(self.current_factor_id, parent_id,[child_id], feature_ids ,z,sigma, idx_map)
        
        self.factors[self.current_factor_id] = factor
        parent = self.pose_nodes[parent_id]
        child = self.pose_nodes[child_id]
        
        features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
        parent.factor[self.current_factor_id] = factor
        child.factor[self.current_factor_id] = factor
        for feature in features:
            feature.factor[self.current_factor_id] = factor
            
        self.current_factor_id += 1    
        
    def remove_factor(self, factor):
        del self.pose_nodes[factor.parent_id].factor[factor.id]
            
        for id_ in factor.children_ids:
            del self.pose_nodes[id_].factor[factor.id]
                
        for id_ in factor.feature_ids:
            del self.feature_nodes[id_].factor[factor.id]
        
        del self.factors[factor.id]

    def prune(self, node_id):
        node = self.pose_nodes[node_id]
        for factor in list(node.factor.values()).copy():
            self.remove_factor(factor)
            
        del self.pose_nodes[node.id]    
        