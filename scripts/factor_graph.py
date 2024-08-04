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
    def __init__(self, id_, parent_node, children_nodes, feature_nodes, z, sigma, idx_map ):
        self.prior = parent_node == None
        self.parent = parent_node
        self.children = children_nodes
        self.feature_nodes = feature_nodes
        self.z = z
        self.omega = inv(sigma)
        self.omega = (self.omega.T+self.omega)/2
        self.pruned = False
        self.n_features = len(feature_nodes)
        self.n_poses =  len(children_nodes)
        self.idx_map = idx_map
        self.id = id_
        
    def copy(self):
        return deepcopy(self)
    
    def get_jacobian(self, M_init):
        pass 
    
    def prune(self, node_id):
        if not self.parent.id == node_id:
            del self.parent.factor[self.id]
        for node in self.children:
            if not node.id == node_id:
                del self.parent.factor[self.id]
                
        for node in self.feature_nodes:
            del self.node.factor[self.id]
            
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
        children = [self.pose_nodes[id_] for id_ in pose_idx_map.keys()]
        features = [self.feature_nodes[id_] for id_ in feature_idx_map.keys()]
                  
        factor = Factor(self.prior_factor.id, None,children,features ,z,sigma, idx_map)  
            
        self.prior_factor = factor
                    
    def add_factor(self, parent_id, child_id, feature_ids, z, sigma, feature_idx_map):
        idx_map={"pose": {child_id: 0}, "features": feature_idx_map}
        parent = self.pose_nodes[parent_id]
        child = self.pose_nodes[child_id]
            
        features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
        factor = Factor(self.current_factor_id, parent,[child],features ,z,sigma, idx_map)
        self.factors[self.current_factor_id] = factor
        
        parent.factor[self.current_factor_id] = factor
        child.factor[self.current_factor_id] = factor
        
        for feature in features:
            feature.factor[self.current_factor_id] = factor
            
        self.current_factor_id += 1    