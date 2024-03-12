# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:02 2024

@author: hibad
"""
import sys
sys.path.append('../')

import numpy as np
from scipy.linalg import solve_triangular
from numpy import sin, cos
from numpy.linalg import inv, norm, lstsq
from Lie import SE3, SE2, SO3, SO2
import pickle 
import matplotlib.pyplot as plt 

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
                self.factor=[]
                    
        class Factor:
            def __init__(self, parent_node, child_node, feature_nodes, z, sigma, idx_map):
                self.parent=parent_node
                self.child=child_node
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
                self.omega=(self.omega.T+self.omega)/2
                self.pruned=False
                self.n = len(feature_nodes)
                self.idx_map = idx_map
            
            def Jacobian(self):
                pass 
            
        def __init__(self):
            self.nodes=[]
            self.pose_nodes={}
            self.factors=[]
            self.feature_nodes={}
            self.window = 20
            self.current_pose_id = -1
            
        def prune_graph(self):
            pass
        
        def add_node(self, M, node_type, feature_id=None, ):
            i=self.current_pose_id+1
            if node_type=="pose":
                node=self.Node(i,M, node_type)
                self.pose_nodes[i]=node
                self.current_pose_id = i
                if len(self.pose_nodes)>=self.window:
                    self.prun_graph()
                    
            if node_type=="feature":
                node=self.Node(feature_id,M, node_type)
                self.feature_nodes[feature_id]=node
            self.nodes.append(node)                
            return self.current_pose_id
        
        def add_factor(self, parent_id, child_id, feature_ids, z, sigma, idx_map):
            if  parent_id == None:
                parent = None
            else:
                parent = self.pose_nodes[parent_id]
                
            if  child_id == None:
                child = None
            else:
                child = self.pose_nodes[child_id]
                
            features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
            self.factors.append(self.Factor(parent,child,features ,z,sigma, idx_map))
                
        
class Back_end:
    # def get_pose_jacobian(self, tau_1, tau_2, z_bar):
    #     J1=-SE3.Jl_inv(z_bar)
    #     J2=SE3.Jr_inv(z_bar)
    #     return J1, J2
    
  
    # def get_feature_jacobian(self, tau_1, tau_2, z_bar):
    #     J1=-SE3.Jl_inv(z_bar)@SE3.Jr(tau_1)
    #     J2=SE3.Jr_inv(z_bar)@SE3.Jr(tau_2)
    #     return J1, J2
    
    def __init__(self):
        pass
    
    def node_to_vector(self, graph):
        self.pose_idx_map={}
        self.feature_idx_map={}
        mu=[]
        for node_id, node in graph.pose_nodes.items():
            if not node.pruned:
                self.pose_idx_map[node_id]=len(mu)*6
                mu.append(node.M)
        for node_id,node in graph.feature_nodes.items():
            self.feature_idx_map[node_id]=len(mu)*6
            mu.append(node.M)
        return mu
    
    def linearize(self,x, factors):
        H = np.zeros((6*len(x), 6*len(x)))
        b = np.zeros(6*len(x))
        for factor in factors:
            idx_map = factor.idx_map.copy()
            omega = factor.omega.copy()
            if not factor.parent == None:
                F = np.zeros((6*len(x), 12+factor.n*6))          #map from factor vector to graph vector
                J = np.zeros((6+factor.n*6,12+factor.n*6)) #map from factor vector to observation
                e = np.zeros((6+factor.n*6)) #difference between observation and expected observation
                 
                idx=self.pose_idx_map[factor.parent.id]
                F[idx:idx+6,0:6] = np.eye(6)
                z = factor.z[0:6].copy()
                M_r1 = factor.parent.M.copy()
                M_r1_inv = inv(M_r1)
                M_r2 = factor.child.M.copy()
                z_bar = SE3.Log(M_r1_inv@M_r2)
                J1=-SE3.Jl_inv(z_bar)
                J2=SE3.Jr_inv(z_bar)
                J[0:6,0:6] = J1
                J[0:6, 6:12] = J2
                e[0:6] = z - z_bar
                
                idx=self.pose_idx_map[factor.child.id]
                F[idx:idx+6,6:12] = np.eye(6)

                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(M_r1_inv@feature.M.copy())

                    J1=-SE3.Jl_inv(z_bar)
                    J2=SE3.Jr_inv(z_bar)
                    J[i:i+6, 0:6] = J1
                    J[i:i+6, 6+i:6+i+6] = J2
                    e[i:i+6] = z - z_bar
                    
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+6,6+i:6+i+6] = np.eye(6)
            else:
                omega=np.eye(6)
                J = np.eye(len(factor.z))
                F = np.zeros((6*len(x), len(factor.z)))   
                e = np.zeros(len(factor.z))
                if not factor.child == None:
                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(factor.child.M.copy())
                    e[0:6] = z - z_bar
                    J[0:6, 0:6] = SE3.Jr_inv(z_bar)
                    idx=self.pose_idx_map[factor.child.id]
                    F[idx:idx+6,0:6] = np.eye(6)
                    
                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    z = factor.z[i:i+6].copy()
                    z_bar = SE3.Log(feature.M.copy())
                    J[i:i+6, i:i+6] = SE3.Jr_inv(z_bar)
                    #e[i:i+6] = z - z_bar
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+6,i:i+6] = np.eye(6)
                #print(e)
                global test
                test=F@J.T@omega@J@F.T
            H+=F@J.T@omega@J@F.T
            b+=F@J.T@omega@e

        return H, b
    
    def linear_solve(self, A,b):
        A=(A+A.T)/2
        # L=np.linalg.cholesky(A)
        # y=solve_triangular(L,b, lower=True)
        
        #return solve_triangular(L.T, y)
        return lstsq(A,b)[0]
    
    def update_nodes(self, graph,dx, cov):
        for node in graph.pose_nodes.values():
            if not node.pruned:
                idx=self.pose_idx_map[node.id]
                node.M=node.M@SE3.Exp(dx[idx:idx+6])
                node.H=cov[idx:idx+6,idx:idx+6].copy()
                
        for node in graph.feature_nodes.values():
            idx=self.feature_idx_map[node.id]
            node.M=node.M@SE3.Exp(dx[idx:idx+6])
            node.H=cov[idx:idx+6,idx:idx+6].copy()

        
    def optimize(self, graph):
        with open('graph.pickle', 'wb') as handle:
            pickle.dump(graph, handle)
        print("optimizing graph")
        x = self.node_to_vector(graph)
        H,b=self.linearize(x,graph.factors)
        dx=self.linear_solve(H,b)
        # x+=dx 
        i=0
        self.update_nodes(graph, 0.05*dx.copy(),np.zeros(H.shape))
        while np.max(np.abs(dx))>0.0001 and i<10000:
            print(i)
            H,b=self.linearize(x,graph.factors)
            global dx_test
            dx_test=dx
            dx=self.linear_solve(H,b)
            # x+= dx
            self.update_nodes(graph, 0.05*dx.copy(),np.zeros(H.shape))
            i+=1


        self.update_nodes(graph, np.zeros(6*len(x)),inv(H))
        print("optimized")

        return x, H
    
solver=Back_end()
with open('graphSE3.pickle', 'rb') as handle:
    graph = pickle.load(handle)

plt.figure(0)
for node in graph.pose_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    print(M)
    plt.plot(M[0,3], M[1,3], "o")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[5]), 0.1*sin(mu[5]))
    
for node in graph.feature_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    print(M)

    plt.plot(M[0,3], M[1,3], "x")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[5]), 0.1*sin(mu[5]))
plt.axis('scaled')
    
x, H = solver.optimize(graph)
#%%
plt.figure(1)
for node in graph.pose_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    print(M)
    plt.plot(M[0,3], M[1,3], "o")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[5]), 0.1*sin(mu[5]))
    
for node in graph.feature_nodes.values():
    M=node.M
    mu=SE3.Log(M)
    print(M)

    plt.plot(M[0,3], M[1,3], "x")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[5]), 0.1*sin(mu[5]))
plt.axis('scaled')

#%%
graph.feature_nodes[12].M@inv(SE3.Exp(graph.factors[1].z[6:12]))
graph.pose_nodes[0].M