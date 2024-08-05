#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:41:45 2023

@author: barc
"""

import numpy as np
from copy import deepcopy
from common_functions import np2pc
# from scipy.linalg import solve_triangular
from numpy.linalg import inv, norm, lstsq
from copy import deepcopy
from Lie import SE3
import open3d as o3d
import pickle
import threading
from factor_graph import Factor_Graph

np.set_printoptions(precision=2)


def update_pose(M, dx):
     M = [m@SE3.Exp(dx[6*i:6*i+6]) if 6*i+6<=len(dx) else m for i, m in enumerate(M) ]
     return M
 
class Graph_SLAM:            
    class Back_end:    
        def __init__(self, max_iter, step_size):
            self.max_iter = max_iter
            self.step_size = step_size
            
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
                for child_id in prior.children_ids:
                    i = 6*prior_idx_map["pose"][child_id]
                    idx = global_idx_map["pose"][child_id]
                    z = prior.z[i:i+6].copy()
                    z_bar = SE3.Log(M[idx])
                    e[i:i+6] = SE3.Log(SE3.Exp(z - z_bar))
                    J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                    F[6*idx:6*idx+6,i:i+6] = np.eye(6)
                    
                if not localize_mode:
                    for feature_id in prior.feature_ids:
                        idx = global_idx_map["features"][feature_id]
                        i = 6*prior_idx_map["features"][feature_id]
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
                idx = global_idx_map['pose'][factor.parent_id]
                F[6*idx:6*idx+6,0:6] = np.eye(6)
                z = factor.z[0:6].copy()
                M_r1 = M[idx]
                M_r1_inv = inv(M_r1)
                
                idx = global_idx_map['pose'][factor.children_ids[0]]
                F[6*idx:6*idx+6,6:12] = np.eye(6)
                M_r2 = M[idx]
                
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J[0:6,0:6] = -SE3.Jl_inv(z_bar)
                J[0:6, 6:12] = SE3.Jr_inv(z_bar)
                e[0:6] = SE3.Log(SE3.Exp(z - z_bar))
                
                #features
                for feature_id in factor.feature_ids:
                    i = 6*factor_idx_map[feature_id]
                    idx = global_idx_map['features'][feature_id]
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
        def linear_solve(A,b):
            if np.linalg.det(A) == 0:
                # A+=np.eye(len(b))*0.0001
                print("sigular")
            # A=(A+A.T)/2
            # L=np.linalg.cholesky(A)
            # y=solve_triangular(L,b, lower=True)
            
            # return solve_triangular(L.T, y)
            return lstsq(A,b)[0]
    
        def optimize(self, graph, localize_mode = False):
            # with open('graph.pickle', 'wb') as handle:
            #     pickle.dump(graph, handle)
            print("optimizing graph")
            M, idx_map = self.node_to_vector(graph)
            prior =  graph.prior_factor.copy()
            factors = graph.factors.copy()
            
            H,b=self.linearize(M.copy(), prior , factors, idx_map, localize_mode)
            dx=self.linear_solve(H,b)
            i=0
            M = update_pose(M, self.step_size*dx)

            while np.max(np.abs(dx))>0.001 and i<self.max_iter:
                print("step: ", i)
                H,b=self.linearize(M.copy(), prior ,factors, idx_map, localize_mode)
                print("solve MAP")
                dx=self.linear_solve(H,b)
                print("update pose")
                M =update_pose(M, self.step_size*dx,)
                # M = self.update_pose(M.copy(), 1*dx)
                i+=1
                print(max(np.abs(dx)))
            # self.update_nodes(graph, M, inv(H), idx_map, localize_mode)
            print("optimized")
            
            return M, H, idx_map
        
        def marginalize(self,graph, node, localize_mode):
            print("marginalize node", node.id)
            prior = graph.prior_factor
            pose_idx_map={}
            feature_idx_map={}
            n = 0
            for key in prior.idx_map["pose"].keys():
                pose_idx_map[key] = n
                n+=1
                
            for factor in node.factor.values():
                if not factor.parent_id in prior.idx_map["pose"].keys():
                    pose_idx_map[factor.parent_id] = n
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
                M[i] = graph.pose_nodes[id_].M.copy()
                
            for id_ , i in feature_idx_map.items():
                M[i] = graph.feature_nodes[id_].M.copy()
                
                
            H, b = Graph_SLAM.Back_end.linearize(M, prior, node.factor, {"pose": pose_idx_map, "features": feature_idx_map}, localize_mode)   
            dx=Graph_SLAM.Back_end.linear_solve(H,b)
            M =  update_pose(M, 0.5*dx)
            i = 0
            while np.max(np.abs(dx))>0.001 and i<1000:
                print("step: ", i)
                H, b = Graph_SLAM.Back_end.linearize(M, prior, node.factor, {"pose": pose_idx_map, "features": feature_idx_map}, localize_mode)   
                dx=Graph_SLAM.Back_end.linear_solve(H,b)

                M =  update_pose(M, 0.5*dx)
                i+=1
                
            cov = inv(H)
           
            node_idx = pose_idx_map[node.id]
            idx_range = np.arange(6*node_idx,6*node_idx+6)
            del pose_idx_map[node.id]
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
            cov = cov + graph.forgetting_factor*np.eye(len(cov))
            
            if localize_mode:
                graph.add_prior_factor(z, cov, pose_idx_map ,{})
            else:
                graph.add_prior_factor(z, cov, pose_idx_map ,feature_idx_map)

        def prune(self, graph, horizon, localize_mode):
            for node in list(graph.pose_nodes.values())[:-horizon]:
                self.marginalize(graph, node, localize_mode)
                graph.prune(node.id)
                
    def __init__(self, M_init, localize_mode = False,horizon = 10 ,forgetting_factor = 0, max_iter=50, step_size = 0.01):
        self.back_end=self.Back_end(max_iter, step_size)
        self.forgetting_factor = forgetting_factor
        self.horizon = horizon
        self.localize_mode = localize_mode
        self.global_map=None
        self.optimized = False
        self.M=M_init.copy()
        self.backend_thread = threading.Thread(target = self.optimize,daemon=True, args = ())
        self.reset()
        self.backend_thread.start()

    def reset(self):
        self.factor_graph=Factor_Graph(self.horizon, self.forgetting_factor)
        self.current_node_id=self.factor_graph.add_node(self.M, "pose")
        self.omega=np.eye(3)*0.001
        self.feature_tree=None
        
        # self.costmap=self.anomaly_detector.costmap
    
    def _posterior_to_factor(self, posterior, local_cloud, key_node):
        mu = posterior["mu"]
        sigma = posterior["sigma"]
        idx_map = posterior["features"]

        self.factor_graph.pose_nodes[self.current_node_id].local_map=local_cloud
        new_node_id=self.factor_graph.add_node(self.M.copy(),"pose", key_node = key_node)

        feature_node_id = idx_map.keys()
        z= np.zeros(6*len(mu))
        J = np.zeros((6*len(mu), 6*len(mu)))
        for i, M in enumerate(mu):
            tau = SE3.Log(M)
            z[6*i:6*i+6] = tau
            J[6*i:6*i+6, 6*i:6*i+6] = SE3.Jr_inv(tau)
        sigma = J@sigma@J.T 
        self.factor_graph.add_factor(self.current_node_id,new_node_id,feature_node_id, z,sigma, idx_map)
        self.current_node_id=new_node_id      
        return new_node_id
    
    def global_map_assemble(self):
        points=[]
        colors=[]
        for node in self.factor_graph.pose_nodes.values():
            if not node.local_map == None:
                if( len(node.local_map["features"])) == 0:
                   M = node.M.copy() 
                else:
                    dm = np.zeros(6)
                    for feature_id, feature in node.local_map["features"].items():
                        dm  += SE3.Log(self.factor_graph.feature_nodes[feature_id].M.copy()@inv(feature['M']))
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
            if not feature_id in self.factor_graph.feature_nodes.keys():
                Z=mu[idx]
                M=M_node@Z
                self.factor_graph.add_node(M,"feature", feature_id)
    
    def get_node_est(self, node_id=-1):
        if node_id==-1:
            node_id = self.current_node_id
        return self.factor_graph.pose_nodes[node_id].M.copy()
    
    def get_features_est(self):
        features = {}
        for id_, node in self.factor_graph.feature_nodes.items():
            features[id_] = node.M
        return features
    
    def update(self, posterior): 
        if self.optimized:
            self.optimized=False
            
        M = self.factor_graph.pose_nodes[self.current_node_id].M.copy()
        U = posterior["mu"][0]        
        self.M = M@U
        self.init_new_features(M, posterior["mu"], posterior["features"])        
        return self.M.copy()
    
    def place_node(self, posterior, local_cloud, key_node = False):
        node_id = self._posterior_to_factor(posterior, local_cloud, key_node)
        self.backend_thread.join()
        self.backend_thread = threading.Thread(target = self.optimize,daemon=True, args = ())
        self.backend_thread.start()
        # with open('graph.pickle', 'wb') as handle:
        #     pickle.dump(self.factor_graph, handle)
        # self.optimize()
        return node_id
    
    def update_nodes(self, M, H, idx_map):
        print("update start")
        if np.linalg.det(H)==0:
            H += np.eye(len(H))*0.001
            
        cov = inv(H)            
        for node_id, idx in idx_map["pose"].items():
            self.factor_graph.pose_nodes[node_id].M = M[idx]
            self.factor_graph.pose_nodes[node_id].cov = cov[6*idx:6*idx+6,6*idx:6*idx+6].copy()
            
        if not self.localize_mode:
            for node_id, idx in idx_map["features"].items():  
                self.factor_graph.feature_nodes[node_id].M = M[idx]
                self.factor_graph.feature_nodes[node_id].cov = cov[6*idx:6*idx+6,6*idx:6*idx+6].copy()
        print("update end")
        
    def optimize(self):
        print("optimizing")
    # with open('graph.pickle', 'wb') as handle:
    #     pickle.dump(self.factor_graph, handle)
    
        M, H, idx_map = self.back_end.optimize(self.factor_graph, self.localize_mode)
    
        print("apply results")
        self.update_nodes(M, H, idx_map)

        self.omega = H
        # self.global_map_assemble()
        self.optimized = True
        print("start prune")
        self.back_end.prune(self.factor_graph, 10, self.localize_mode)
        print("optimize end")

if __name__ == "__main__":
    graph_slam = Graph_SLAM(np.zeros(4), True, 1, 0, 1000)
    with open('tests/graph.pickle', 'rb') as f:
        graph = pickle.load(f)
    graph_slam.factor_graph = graph
    graph_slam.factor_graph.prune(5, True)
    graph_slam.back_end.optimize(graph_slam.factor_graph,  localize_mode = True)