# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:14:24 2024

@author: hibad
"""

# import sys
# sys.path.append('../')

import numpy as np
from scipy.linalg import solve_triangular
from numpy import sin, cos
from numpy.linalg import inv, norm, lstsq
from Lie import SE3, SE2, SO3, SO2
import pickle 
import matplotlib.pyplot as plt 
import time 
from copy import deepcopy
np.set_printoptions(precision=2)

class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, M, node_type):
                self.type=node_type
                self.M=M
                self.H=np.zeros((6,6))
                self.id=node_id
                self.local_map=None
                self.pruned=False 
                self.depth_img=None
                self.factor={}
               # self.prior_factor = None
                    
        class Factor:
            def __init__(self, id_, parent_node, children_nodes, feature_nodes, z, sigma, idx_map ):
                self.parent=parent_node
                self.children=children_nodes
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
                self.omega=(self.omega.T+self.omega)/2
                self.pruned=False
                self.n = len(feature_nodes)
                self.idx_map = idx_map
                self.id = id_
                
            def get_jacobian(self):
                pass 
            
            def prune(self, node_id):
                if not self.parent.id == node_id:
                    self.parent.factor.pop(self.id)
                for node in self.children:
                    if not node.id == node_id:
                        node.factor.pop(self.id)
                        
                for node in self.feature_nodes:
                    node.factor.pop(self.id)
                    
        def __init__(self):
            self.prior_factor = None
            self.pose_nodes={}
            self.factors={}
            self.feature_nodes={}
            self.window = 1
            self.current_pose_id = -1
            self.current_factor_id = 0
        
        def marginalize(self, node):
            prior = self.prior_factor
            pose_idx_map=prior.idx_map["pose"].copy()
            feature_idx_map=prior.idx_map["features"].copy()
            n = len(pose_idx_map) + len(feature_idx_map)
                            
            for factor in node.factor.values():
                if not factor.parent.id in factor.idx_map["pose"].keys():
                    pose_idx_map[factor.parent.id] = n*6
                    n += 1      
                    
                for id_  in factor.idx_map["pose"].keys():
                    print("factor id: ", id_)
                    if not id_ in pose_idx_map.keys():
                        pose_idx_map[id_] = n*6
                        n += 1
                for id_ in factor.idx_map["features"].keys():
                    if not id_ in feature_idx_map.keys():
                        feature_idx_map[id_] = n*6
                        n += 1
                    
            H=np.zeros((6*n,6*n)) 
            b = np.zeros(6*n)
            
            H[0:len(prior.omega),0:len(prior.omega)] = prior.omega
            b[0:len(prior.omega)] = prior.omega@prior.z
            
            for id_, factor in node.factor.items():
                J = np.zeros((12+factor.n, 12+factor.n))
                idx_map = factor.idx_map["features"].copy()
                F = np.zeros((6*n, 12+factor.n*6))          #map from factor vector to graph vector
                J = np.zeros((6+factor.n*6,12+factor.n*6)) #map from factor vector to observation
                 
                idx=pose_idx_map[factor.parent.id]
                F[idx:idx+6,0:6] = np.eye(6)
                M_r1 = factor.parent.M.copy()
                M_r1_inv = inv(M_r1)
                M_r2 = factor.children[0].M.copy()
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)@SE3.Jr(SE3.Log(M_r1))
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)@SE3.Jr(SE3.Log(M_r2))
                
                idx=pose_idx_map[factor.children[0].id]
                F[idx:idx+6,6:12] = np.eye(6)
    
                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    M_f = feature.M.copy()
                    z_bar = SE3.Log(M_r1_inv@M_f)
    
                    J[i:i+6, 0:6] = -SE3.Jl_inv(z_bar)@SE3.Jr(SE3.Log(M_r1))
                    J[i:i+6, 6+i:6+i+6] = SE3.Jr_inv(z_bar)@SE3.Jr(SE3.Log(M_f))
                    
                    idx = feature_idx_map[feature.id]
                    F[idx:idx+6,6+i:6+i+6] = np.eye(6)
            
              
                H += F@(J.T@factor.omega@J)@F.T
                b += F@J.T@factor.omega@factor.z
            cov = inv(H)
            z = cov@b
            
            node_idx = pose_idx_map[node.id]
            idx_range = np.arange(node_idx,node_idx+6)
           
            pose_idx_map.pop(node.id)
            
            for key, idx in pose_idx_map.items():
                if idx > node_idx:
                    pose_idx_map[key] -=6
                    
            for key, idx in feature_idx_map.items():
                if idx > node_idx:
                    feature_idx_map[key] -=6
                    
            prior.z = np.delete(z,idx_range, 0)
            cov = np.delete(H,idx_range, 0 )
            cov = np.delete(cov,idx_range, 1 )
        
            prior.omega = inv(cov)
            prior.omega = (prior.omega + prior.omega.T)/2
            prior.idx_map={"features": feature_idx_map, "pose": pose_idx_map}    
            prior.children = [self.pose_nodes[id_] for id_ in pose_idx_map.keys()]
            prior.feature_nodes = [self.feature_nodes[id_] for id_ in feature_idx_map.keys()]

            self.prior_factor = prior
            
        def prune(self):
            for node in list(self.pose_nodes.values())[:-self.window]:
                self.marginalize(node)
                for id_, factor in node.factor.items():
                    factor.prune(node.id)
                    self.factors.pop(id_)
                self.pose_nodes.pop(node.id)
    
        def add_node(self, M, node_type, feature_id=None, ):
            i=self.current_pose_id+1
            if node_type=="pose":
                node=self.Node(i,M, node_type)
                self.pose_nodes[i]=node
                self.current_pose_id = i
                    
            if node_type=="feature":
                node=self.Node(feature_id,M, node_type)
                self.feature_nodes[feature_id]=node
            return self.current_pose_id
        
        def add_prior_factor(self, children_ids, features_ids, z, sigma, pose_idx_map ,feature_idx_map):
            idx_map={"pose": pose_idx_map, "features": feature_idx_map}
            children = [self.pose_nodes[id_] for id_ in children_ids]
            features = [self.feature_nodes[id_] for id_ in features_ids]
                      
            factor = self.Factor(self.current_factor_id, None,children,features ,z,sigma, idx_map)  
                
            self.prior_factor = factor
                        
        def add_factor(self, parent_id, child_id, feature_ids, z, sigma, feature_idx_map):
            idx_map={"pose": {child_id: 0}, "features": feature_idx_map}
            parent = self.pose_nodes[parent_id]
            child = self.pose_nodes[child_id]
                
            features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
            factor = self.Factor(self.current_factor_id, parent,[child],features ,z,sigma, idx_map)
            self.factors[self.current_factor_id] = factor
            
            parent.factor[self.current_factor_id] = factor
            child.factor[self.current_factor_id] = factor
            
            for feature in features:
                feature.factor[self.current_factor_id] = factor
                
            self.current_factor_id += 1    
            
    class Back_end:    
        def __init__(self):
            pass
        
        def node_to_vector(self, graph):
            self.pose_idx_map={}
            self.feature_idx_map={}
            n = 0 
            for node_id, node in graph.pose_nodes.items():
                if not node.pruned:
                    self.pose_idx_map[node_id]=n*6
                    n+=1
                    
            for node_id,node in graph.feature_nodes.items():
                self.feature_idx_map[node_id]=n*6
                n+=1
    
            return n
        
        def linearize(self,n, prior, factors):
            H = np.zeros((6*n, 6*n))
            b = np.zeros(6*n)
            
            J = np.eye(len(prior.z))
            F = np.zeros((6*n, len(prior.z)))   
            e = np.zeros(len(prior.z))
            
            #prior
            idx_map = prior.idx_map["features"].copy()
            pose_idx_map = prior.idx_map["pose"].copy()
            print("prior features", idx_map)
            print("prior pose", pose_idx_map)

            omega = prior.omega.copy()
            for child in prior.children:
                i = pose_idx_map[child.id]
                z = prior.z[i:i+6].copy()
                z_bar = SE3.Log(child.M.copy())
                e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                idx=self.pose_idx_map[child.id]
                F[idx:idx+6,i:i+6] = np.eye(6)
                
            for feature in prior.feature_nodes:
                i = idx_map[feature.id]
                z = prior.z[i:i+6].copy()
                z_bar = SE3.Log(feature.M.copy())
                J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                idx=self.feature_idx_map[feature.id]
                F[idx:idx+6,i:i+6] = np.eye(6)
                
            H+=F@(J.T@omega@J)@F.T
            b+=F@J.T@omega@e    
            for factor in factors.values():
                idx_map = factor.idx_map["features"].copy()
                omega = factor.omega.copy()
                F = np.zeros((6*n, 12+factor.n*6))          #map from factor vector to graph vector
                J = np.zeros((6+factor.n*6,12+factor.n*6)) #map from factor vector to observation
                e = np.zeros(len(factor.z)) #difference between observation and expected observation
                 
                idx=self.pose_idx_map[factor.parent.id]
                F[idx:idx+6,0:6] = np.eye(6)
                z = factor.z[0:6].copy()
                M_r1 = factor.parent.M.copy()
                M_r1_inv = inv(M_r1)
                M_r2 = factor.children[0].M.copy()
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)
                e[0:6] = SE3.Log(SE3.Exp(z - z_bar))
                
                idx=self.pose_idx_map[factor.children[0].id]
                F[idx:idx+6,6:12] = np.eye(6)

                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(M_r1_inv@feature.M.copy())

                    J[i:i+6, 0:6] = -SE3.Jl_inv(z_bar)
                    J[i:i+6, 6+i:6+i+6] = SE3.Jr_inv(z_bar)
                    
                    e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+6,6+i:6+i+6] = np.eye(6)
            
              
                H+=F@(J.T@omega@J)@F.T
                b+=F@J.T@omega@e
            return H, b
        
        def linear_solve(self, A,b):
            # A=(A+A.T)/2
            # L=np.linalg.cholesky(A)
            # y=solve_triangular(L,b, lower=True)
            
            # return solve_triangular(L.T, y)
            return lstsq(A,b)[0]
        
        def update_nodes_pose(self, graph,dx):
            for node_id, idx in self.pose_idx_map.items():
                graph.pose_nodes[node_id].M = graph.pose_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
      
            for node_id, idx in self.feature_idx_map.items():
                graph.feature_nodes[node_id].M = graph.feature_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
    
        def update_nodes_cov(self, graph,cov):
            for node_id, idx in self.pose_idx_map.items():
                graph.pose_nodes[node_id].cov = cov[idx:idx+6,idx:idx+6].copy()
            for node_id, idx in self.feature_idx_map.items():    
                graph.feature_nodes[node_id].cov = cov[idx:idx+6,idx:idx+6].copy()

        def optimize(self, graph):
            # with open('graph.pickle', 'wb') as handle:
            #     pickle.dump(graph, handle)
            print("optimizing graph")
            n = self.node_to_vector(graph)
            H,b=self.linearize(n, graph.prior_factor , graph.factors)
            dx=self.linear_solve(H,b)
            i=0
            self.update_nodes_pose(graph, 1*dx.copy())
            while np.max(np.abs(dx))>0.01 and i<10:
                print(i)
                H,b=self.linearize(n, graph.prior_factor , graph.factors)
                dx=self.linear_solve(H,b)
                self.update_nodes_pose(graph, 1*dx.copy())
                i+=1
    
            self.update_nodes_cov(graph, inv(H))
            print("optimized")
            
            with open('graph.pickle', 'wb') as handle:
                graph_test = deepcopy(graph)
                for node in graph_test.pose_nodes.values():
                    node.local_map = None
                pickle.dump(graph_test, handle)
            return H
with open('graph.pickle', 'rb') as handle:
    graph = pickle.load(handle)

graph_prune = deepcopy(graph)
graph_prune.prune()
prior = graph_prune.prior_factor

optimizer = Graph_SLAM.Back_end()
optimizer.optimize(graph_prune)