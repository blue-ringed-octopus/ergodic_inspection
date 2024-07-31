#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:41:45 2023

@author: barc
"""

import numpy as np

from common_functions import np2pc
# from scipy.linalg import solve_triangular
from numpy.linalg import inv, norm, lstsq
from copy import deepcopy
from Lie import SE3
import open3d as o3d
import pickle

np.set_printoptions(precision=2)

class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, M, node_type):
                self.type=node_type
                self.M=M
                self.cov=np.zeros((6,6))
                self.id=node_id
                self.local_map=None
                self.pruned=False 
                self.factor={}
               # self.prior_factor = None
                    
        class Factor:
            def __init__(self, id_, parent_node, children_nodes, feature_nodes, z, sigma, idx_map ):
                self.prior = parent_node == None
                self.parent=parent_node
                self.children=children_nodes
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
                self.omega=(self.omega.T+self.omega)/2
                self.pruned=False
                self.n_features = len(feature_nodes)
                self.n_poses =  len(children_nodes)
                self.idx_map = idx_map
                self.id = id_
                
            def get_jacobian(self, M_init):
                pass 
            
            def prune(self, node_id):
                if not self.parent.id == node_id:
                    self.parent.factor.pop(self.id)
                for node in self.children:
                    if not node.id == node_id:
                        node.factor.pop(self.id)
                        
                for node in self.feature_nodes:
                    node.factor.pop(self.id)
                    
        def __init__(self, forgetting_factor):
            self.prior_factor = self.Factor(0, [],[], [], None, np.eye(2),  {"features":{}, "pose":{}})
            self.pose_nodes={}
            self.factors={}
            self.feature_nodes={}
            self.window = 10
            self.current_pose_id = -1
            self.current_factor_id = 1
            self.forgetting_factor = forgetting_factor
        
        def marginalize(self, node, localize_mode):
            print("marginalize node", node.id)
            prior = self.prior_factor
            pose_idx_map={}
            feature_idx_map={}
            n = 0
            for key in prior.idx_map["pose"].keys():
                pose_idx_map[key] = n
                n+=1
                
            for factor in node.factor.values():
                if not factor.parent.id in prior.idx_map["pose"].keys():
                    pose_idx_map[factor.parent.id] = n
                    n += 1      
                    
                for id_  in factor.idx_map["pose"].keys():
                    if not id_ in pose_idx_map.keys():
                        pose_idx_map[id_] = n
                        n += 1
                        
            for key in prior.idx_map["features"].keys():
                feature_idx_map[key] = n
                n+=1       
                
            for factor in node.factor.values():                        
                for id_ in factor.idx_map["features"].keys():
                    if not id_ in feature_idx_map.keys():
                        feature_idx_map[id_] = n
                        n += 1
                        
            M = np.zeros((n, 4,4))   
            for id_ , i in pose_idx_map.items():
                M[i] = self.pose_nodes[id_].M.copy()
                
            for id_ , i in feature_idx_map.items():
                M[i] = self.feature_nodes[id_].M.copy()
                
                
            H, b = Graph_SLAM.Back_end.linearize(M, prior, node.factor, {"pose": pose_idx_map, "features": feature_idx_map}, localize_mode)   
            dx=Graph_SLAM.Back_end.linear_solve(H,b)
            M =  Graph_SLAM.Back_end.update_pose(M, 0.5*dx)
            i = 0
            while np.max(np.abs(dx))>0.0001 and i<100:
                print("step: ", i)
                H, b = Graph_SLAM.Back_end.linearize(M, prior, node.factor, {"pose": pose_idx_map, "features": feature_idx_map}, localize_mode)   
                dx=Graph_SLAM.Back_end.linear_solve(H,b)

                M =  Graph_SLAM.Back_end.update_pose(M, 0.5*dx)
                i+=1
                
            cov = inv(H)
           
            node_idx = pose_idx_map[node.id]
            idx_range = np.arange(6*node_idx,6*node_idx+6)
            pose_idx_map.pop(node.id)
            
            for key, idx in pose_idx_map.items():
                if idx > node_idx:
                    pose_idx_map[key] -=1
                
            for key, idx in feature_idx_map.items():
                if idx > node_idx:
                    feature_idx_map[key] -=1
                    
            n_prior = len(pose_idx_map)
            if not localize_mode:
                n_prior += len(feature_idx_map)
            M = np.delete(M,node_idx,0) 
            
            z = np.concatenate([SE3.Log(M[i]) for  i in range(n_prior)])
            cov = np.delete(cov,idx_range, 0 )
            cov = np.delete(cov,idx_range, 1 )
            
            J = np.zeros(((len(z)), len(z)))
            
            for i in range(n_prior):
                J[6*i:6*i+6, 6*i:6*i+6] = SE3.Jr_inv(z[6*i:6*i+6])
            
            cov = J@cov@J.T 
            cov = cov + self.forgetting_factor*np.eye(len(cov))
            children = [self.pose_nodes[id_] for id_ in pose_idx_map.keys()]
            if localize_mode:
                idx_map={"features": {}, "pose": pose_idx_map}    
                feature_nodes = []
            else:
                idx_map={"features": feature_idx_map, "pose": pose_idx_map}    
                feature_nodes = [self.feature_nodes[id_] for id_ in feature_idx_map.keys()]
            
            prior =  self.Factor(self.prior_factor.id, None, children, feature_nodes, z, cov, idx_map)
            self.prior_factor = prior
            
        def prune(self, window, localize_mode):
            for node in list(self.pose_nodes.values())[:-window]:
                self.marginalize(node, localize_mode)
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
                      
            factor = self.Factor(self.prior_factor.id, None,children,features ,z,sigma, idx_map)  
                
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
        
        @staticmethod
        def node_to_vector(graph):
            pose_idx_map={}
            feature_idx_map={}
            n = 0 
            M = []
            for node_id, node in graph.pose_nodes.items():
                if not node.pruned:
                    M.append(node.M.copy())
                    pose_idx_map[node_id]=n
                    n+=1
            
            for node_id,node in graph.feature_nodes.items():
                M.append(node.M.copy())
                feature_idx_map[node_id]=n
                n+=1
    
            return M, {"pose":pose_idx_map, "features": feature_idx_map }
        
        @staticmethod
        def linearize(M, prior, factors, global_idx_map, localize_mode = False):
            if localize_mode:
                n_global = len(global_idx_map["pose"])
            else:
                n_global = len(M)
                
            H = np.zeros((6*n_global, 6*n_global))
            b = np.zeros(6*n_global)
            
            #prior
            if localize_mode:
                n_prior = (prior.n_poses)
            else:
                n_prior = (prior.n_features+prior.n_poses)
            
            if n_prior>0:
                J = np.eye(6*n_prior)
                F = np.zeros((6*n_global, 6*n_prior))   
                e = np.zeros(6*n_prior)
                prior_idx_map = prior.idx_map.copy()
                omega = prior.omega.copy()
                for child in prior.children:
                    i = 6*prior_idx_map["pose"][child.id]
                    idx = global_idx_map["pose"][child.id]
                    z = prior.z[i:i+6].copy()
                    z_bar = SE3.Log(M[idx])
                    e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                    J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                    F[6*idx:6*idx+6,i:i+6] = np.eye(6)
                    
                if not localize_mode:
                    for feature in prior.feature_nodes:
                        idx = global_idx_map["features"][feature.id]
                        i = 6*prior_idx_map["features"][feature.id]
                        F[6*idx:6*idx+6,i:i+6] = np.eye(6)
                        z = prior.z[i:i+6].copy()
                        z_bar = SE3.Log(M[idx])
                        J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                        e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                H+=F@(J.T@omega@J)@F.T
                b+=F@J.T@omega@e    
            
            for factor in factors.values():
                factor_idx_map = factor.idx_map["features"].copy()
                omega = factor.omega.copy()
                n_obsv = 1 + factor.n_features          #child and observed features
                if localize_mode:
                    n_factor = 2               #parent, child and observed features
                else:
                    n_factor = 1 + n_obsv 
                F = np.zeros((6*n_global, 6*n_factor))  #map from factor vector to graph vector
                J = np.zeros((6*n_obsv, 6*n_factor))    #map from factor vector to observation
                e = np.zeros(6*n_obsv)                  #difference between observation and expected observation
                
                #Odometry
                idx = global_idx_map['pose'][factor.parent.id]
                F[6*idx:6*idx+6,0:6] = np.eye(6)
                z = factor.z[0:6].copy()
                M_r1 = M[idx]
                M_r1_inv = inv(M_r1)
                
                idx = global_idx_map['pose'][factor.children[0].id]
                F[6*idx:6*idx+6,6:12] = np.eye(6)
                M_r2 = M[idx]
                
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)
                e[0:6] = SE3.Log(SE3.Exp(z - z_bar))
                
                #features
                for feature in factor.feature_nodes:
                    i = 6*factor_idx_map[feature.id]
                    idx = global_idx_map['features'][feature.id]
                    if not localize_mode:
                        F[6*idx:6*idx+6,6+i:6+i+6] = np.eye(6)

                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(M_r1_inv@M[idx])

                    J[i:i+6, 0:6] = -SE3.Jl_inv(z_bar) #observer jacobian
                    if not localize_mode:
                        J[i:i+6, 6+i:6+i+6] = SE3.Jr_inv(z_bar) #observed jacobian
                    
                    e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
 
                H+=F@(J.T@omega@J)@F.T
                b+=F@J.T@omega@e
            return H, b
        
        @staticmethod
        def linear_solve( A,b):
            # A=(A+A.T)/2
            # L=np.linalg.cholesky(A)
            # y=solve_triangular(L,b, lower=True)
            
            # return solve_triangular(L.T, y)
            return lstsq(A,b)[0]
        
        # def update_nodes_pose(self, graph,dx):
        #     for node_id, idx in self.pose_idx_map.items():
        #         graph.pose_nodes[node_id].M = graph.pose_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
      
        #     for node_id, idx in self.feature_idx_map.items():
        #         graph.feature_nodes[node_id].M = graph.feature_nodes[node_id].M@SE3.Exp(dx[idx:idx+6])
    
        def update_nodes(self, graph, M,cov, idx_map, localize_mode):
            for node_id, idx in idx_map["pose"].items():
                graph.pose_nodes[node_id].M = M[idx]
                graph.pose_nodes[node_id].cov = cov[6*idx:6*idx+6,6*idx:6*idx+6].copy()
                
            if not localize_mode:
                for node_id, idx in idx_map["features"].items():  
                    graph.feature_nodes[node_id].M = M[idx]
                    graph.feature_nodes[node_id].cov = cov[6*idx:6*idx+6,6*idx:6*idx+6].copy()

        @staticmethod
        def update_pose(M, dx):
            M = [m@SE3.Exp(dx[6*i:6*i+6]) if 6*i+6< len(dx) else m for i, m in enumerate(M) ]
            return M
        
        def optimize(self, graph, localize_mode = False):
            # with open('graph.pickle', 'wb') as handle:
            #     pickle.dump(graph, handle)
            print("optimizing graph")
            M, idx_map = self.node_to_vector(graph)
            H,b=self.linearize(M.copy(), graph.prior_factor , graph.factors, idx_map, localize_mode)
            dx=self.linear_solve(H,b)
            i=0
            M = self.update_pose(M, 0.01*dx)

            while np.max(np.abs(dx))>0.001 and i<50:
                print("step: ", i)
                H,b=self.linearize(M.copy(), graph.prior_factor , graph.factors, idx_map, localize_mode)
                dx=self.linear_solve(H,b)

                M = self.update_pose(M, 0.01*dx)
                i+=1
    
            self.update_nodes(graph, M, inv(H), idx_map, localize_mode)
            print("optimized")
            
            # with open('graph.pickle', 'wb') as handle:
            #     graph_test = deepcopy(graph)
            #     pickle.dump(graph_test, handle)
            return H
            
    def __init__(self, M_init, localize_mode = False, forgetting_factor = 0):
        self.forgetting_factor = forgetting_factor
        self.localize_mode = localize_mode
        self.global_map=None
        self.optimized = False
        self.M=M_init.copy()
        self.reset()

    def reset(self):
        self.front_end=self.Front_end(self.forgetting_factor)
        self.back_end=self.Back_end()
        self.current_node_id=self.front_end.add_node(self.M, "pose")
        self.omega=np.eye(3)*0.001
        self.feature_tree=None
        
        # self.costmap=self.anomaly_detector.costmap
    
    def _posterior_to_factor(self, posterior, local_cloud):
        mu = posterior["mu"]
        sigma = posterior["sigma"]
        idx_map = posterior["features"]

        self.front_end.pose_nodes[self.current_node_id].local_map=local_cloud
        new_node_id=self.front_end.add_node(self.M.copy(),"pose")

        feature_node_id = idx_map.keys()
        z= np.zeros(6*len(mu))
        J = np.zeros((6*len(mu), 6*len(mu)))
        for i, M in enumerate(mu):
            tau = SE3.Log(M)
            z[6*i:6*i+6] = tau
            J[6*i:6*i+6, 6*i:6*i+6] = SE3.Jr_inv(tau)
        sigma = J@sigma@J.T 
        self.front_end.add_factor(self.current_node_id,new_node_id,feature_node_id, z,sigma, idx_map)
        self.current_node_id=new_node_id      
        return new_node_id
    
    def global_map_assemble(self):
        points=[]
        colors=[]
        for node in self.front_end.pose_nodes.values():
            if not node.local_map == None:
                if( len(node.local_map["features"])) == 0:
                   M = node.M.copy() 
                else:
                    dm = np.zeros(6)
                    for feature_id, feature in node.local_map["features"].items():
                        dm  += SE3.Log(self.front_end.feature_nodes[feature_id].M.copy()@inv(feature['M']))
                    dm /= len(node.local_map["features"])
                    M = SE3.Exp(dm)
                    
                    cloud = node.local_map['pc'].copy()
                    p = o3d.geometry.PointCloud()
                    p.points=o3d.utility.Vector3dVector(cloud["points"])
                 #   p.colors=o3d.utility.Vector3dVector(cloud["colors"])
                    p=p.transform(M)
                    points.append(np.array(p.points))
                 #   colors.append(np.array(p.colors))
                    colors.append(cloud["colors"])
        if len(points)>0: 
            points=np.concatenate(points)  
            colors=np.concatenate(colors)  
    
            self.global_map = np2pc(points, colors)
            self.global_map = self.global_map.voxel_down_sample(0.05)
            
        return self.global_map
    
    def update_costmap(self, costmap):
        pass 
    
    def init_new_features(self, M_node, mu, features):
        for feature_id, idx in features.items():
            if not feature_id in self.front_end.feature_nodes.keys():
                Z=mu[idx]
                M=M_node@Z
                self.front_end.add_node(M,"feature", feature_id)
    
    def get_node_est(self, node_id=-1):
        if node_id==-1:
            node_id = self.current_node_id
        return self.front_end.pose_nodes[node_id].M.copy()
    
    def get_features_est(self):
        features = {}
        for id_, node in self.front_end.feature_nodes.items():
            features[id_] = node.M
        return features
    
    def update(self, posterior): 
        if self.optimized:
            self.optimized=False
            
        M = self.front_end.pose_nodes[self.current_node_id].M.copy()
        U = posterior["mu"][0]        
        self.M = M@U
        self.init_new_features(M, posterior["mu"], posterior["features"])        
        return self.M.copy()
    
    def place_node(self, posterior, local_cloud, key_node = False):
        node_id = self._posterior_to_factor(posterior, local_cloud)
        H = self.back_end.optimize(self.front_end, self.localize_mode)
        with open('graph.pickle', 'wb') as handle:
            pickle.dump(self.front_end, handle)
        self.omega = H
        # self.global_map_assemble()
        self.optimized = True
        self.front_end.prune(10, self.localize_mode)
        return node_id