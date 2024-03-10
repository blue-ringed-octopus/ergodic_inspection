# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:02 2024

@author: hibad
"""
import sys
sys.path.append('../')

import numpy as np
from scipy.linalg import solve_triangular
from numpy import sin, cos, arctan2
from numpy.linalg import inv, norm, lstsq
from copy import deepcopy
from Lie import SE3, SE2, SO3, SO2
import pickle 
import matplotlib.pyplot as plt 

np.set_printoptions(precision=2)
fr = np.zeros((6,3))
fr[0,0]=1
fr[1,1]=1
fr[5,2]=1
ftag = np.zeros((6,4))
ftag[0,0]=1
ftag[1,1]=1
ftag[2,2] = 1
ftag[5,3] = 1
class Graph_SLAM:
    class Front_end:
        class Node:
            def __init__(self, node_id, mu, node_type):
                self.type=node_type
                self.set_mu(mu)
                self.Cov=np.eye(3)*9999999
                self.id=node_id
                self.local_map=None
                self.pruned=False 
                self.depth_img=None
                self.factor=[]
                
            def set_mu(self,mu):
                self.n=len(mu)
                if self.type == "pose":
                    self.M=SE3.Exp([mu[0], mu[1], 0,0,0, mu[2]])
                    self.mu = fr.T@SE3.Log(self.M)
                else:
                    self.M=SE3.Exp([mu[0], mu[1], mu[2],0,0, mu[3]])
                    self.mu = ftag.T@SE3.Log(self.M)
    
                    
        class Factor:
            def __init__(self, parent_node, child_node, feature_nodes, z, sigma, idx_map):
                self.parent=parent_node
                self.child=child_node
                self.feature_nodes=feature_nodes
                self.z=z
                self.omega=inv(sigma)
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
        
        def add_node(self, x, node_type, feature_id=None, ):
            i=self.current_pose_id+1
            if node_type=="pose":
                node=self.Node(i,x, node_type)
                self.pose_nodes[i]=node
                self.current_pose_id = i
                if len(self.pose_nodes)>=self.window:
                    self.prun_graph()
                    
            if node_type=="feature":
                node=self.Node(feature_id,x, node_type)
                self.feature_nodes[feature_id]=node
            self.nodes.append(node)                
            return self.current_pose_id
        
        def add_factor(self, parent_id, child_id, feature_ids, Z, sigma, idx_map):
            if  parent_id == None:
                parent = None
            else:
                parent = self.pose_nodes[parent_id]
                
            if  child_id == None:
                child = None
            else:
                child = self.pose_nodes[child_id]
                
            features=[self.feature_nodes[feature_id] for feature_id in feature_ids]
            self.factors.append(self.Factor(parent,child,features ,Z,sigma, idx_map))
        
class Back_end:
    def get_pose_jacobian(self, tau_1, tau_2, z_bar):
        J1=-fr.T@SE3.Jl_inv(z_bar)@SE3.Jr(tau_1)@fr
        J2=fr.T@SE3.Jr_inv(z_bar)@SE3.Jr(tau_2)@fr
        return J1, J2
    
  
    def get_feature_jacobian(self, tau_1, tau_2, z_bar):
        J1=-ftag.T@SE3.Jl_inv(z_bar)@SE3.Jr(tau_1)@fr
        J2=ftag.T@SE3.Jr_inv(z_bar)@SE3.Jr(tau_2)@ftag
        return J1, J2
    
    def __init__(self):
        pass
    
    def node_to_vector(self, graph):
        self.pose_idx_map={}
        self.feature_idx_map={}
        x=[]
        for node_id, node in graph.pose_nodes.items():
            if not node.pruned:
                self.pose_idx_map[node_id]=len(x)
                x=np.concatenate((x, node.mu.copy()))
        for node_id,node in graph.feature_nodes.items():
            self.feature_idx_map[node_id]=len(x)
            x=np.concatenate((x, node.mu.copy()))
                
        return np.array(x)
    
    def linearize(self,x, factors):
        H = np.zeros((len(x), len(x)))
        b = np.zeros(len(x))
        for factor in factors:
            idx_map = factor.idx_map.copy()
            omega = factor.omega.copy()
            if not factor.parent == None:
                F = np.zeros((len(x), 6+factor.n*4))         
                J = np.zeros((3+factor.n*4,6+factor.n*4))
                e = np.zeros((3+factor.n*4))
                 
                idx=self.pose_idx_map[factor.parent.id]
                F[idx:idx+3,0:3] = np.eye(3)
                M_r1_inv = inv(factor.parent.M.copy())
                z = factor.z[0:3].copy()
                tau_r1 = fr@factor.parent.mu.copy()
                
                tau_r2 = fr@factor.child.mu.copy()
                z_bar = SE3.Log(M_r1_inv@factor.child.M.copy())
                J1,J2 = self.get_pose_jacobian(tau_r1, tau_r2, z_bar)
                J[0:3,0:3] = J1
                J[0:3, 3:6] = J2
                e[0:3] = z - fr.T@z_bar
                
                idx=self.pose_idx_map[factor.child.id]
                F[idx:idx+3,3:6] = np.eye(3)

                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    tau_tag = ftag@feature.mu.copy()
                    z = factor.z[i:i+4].copy()
                    z_bar = SE3.Log(M_r1_inv@feature.M.copy())

                    J1,J2 = self.get_feature_jacobian(tau_r1, tau_tag, z_bar)
                    J[i:i+4, 0:3] = J1
                    J[i:i+4, 3+i:3+i+4] = J2
                    e[i:i+4] = z - ftag.T@z_bar
                    
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+4,3+i:3+i+4] = np.eye(4)
            else:
                J = np.eye(len(factor.z))
                F = np.zeros((len(x), len(factor.z)))   
                e = np.zeros(len(factor.z))
                if not factor.child == None:
                    z = factor.z[i:i+3].copy()
                    e[0:3] = z - factor.child.mu.copy()
                    idx=self.pose_idx_map[factor.child.id]
                    F[idx:idx+3,0:3] = np.eye(3)
                    
                for feature in factor.feature_nodes:
                    i = idx_map[feature.id]
                    z = factor.z[i:i+4].copy()
                    z_bar = feature.mu.copy()
                    e[i:i+4] = z - z_bar
                    
                    idx=self.feature_idx_map[feature.id]
                    F[idx:idx+4,i:i+4] = np.eye(4)


            H+=F@J.T@omega@J@F.T
            # print("J", J)
            # print(np.max(omega-omega.T))
            # print(np.max((J.T)@omega@J - ((J.T)@omega@J).T))
            # print("omega", omega)
            # print("omega eig", np.min(np.linalg.eig(omega)[0]))
            # print("dh", F@J.T@omega@J@F.T)
            # print("dh eig", np.min(np.linalg.eig(J.T@omega@J)[0]))
            # print("F", F)
            b+=F@J.T@omega@e

        print("H eig", np.min(np.linalg.eig(H)[0]))
        return H, b
    
    def linear_solve(self, A,b):
        A=(A+A.T)/2
        # L=np.linalg.cholesky(A)
        # y=solve_triangular(L,b, lower=True)
        
        #return solve_triangular(L.T, y)
        return lstsq(A,b)[0]
    
    def update_nodes(self, graph,x, cov):
        for node in graph.pose_nodes.values():
            if not node.pruned:
                idx=self.pose_idx_map[node.id]
                nodex=x[idx:idx+node.n]
                nodeCov=cov[idx:idx+node.n,idx:idx+node.n]
                node.set_mu(nodex.copy())
                node.H=nodeCov.copy()
                
        for node in graph.feature_nodes.values():
            idx=self.feature_idx_map[node.id]
            nodex=x[idx:idx+node.n]
            nodeCov=cov[idx:idx+node.n,idx:idx+node.n]
            node.set_mu(nodex.copy())
            node.H=nodeCov.copy()

        
    def optimize(self, graph):
        with open('graph.pickle', 'wb') as handle:
            pickle.dump(graph, handle)
        print("optimizing graph")
        x = self.node_to_vector(graph)
        H,b=self.linearize(x,graph.factors)
        dx=self.linear_solve(H,b)
        x+=dx
        i=0
        self.update_nodes(graph, x,np.zeros(H.shape))
        while np.max(np.abs(dx))>0.0001 and i<10000:
            H,b=self.linearize(x,graph.factors)

            dx=self.linear_solve(H,b)
            print(dx)
            x+=dx
            i+=1
            self.update_nodes(graph, x,np.zeros(H.shape))

        self.update_nodes(graph, x,inv(H))
        print("optimized")

        return x, H
    
solver=Back_end()
with open('graph.pickle', 'rb') as handle:
    graph = pickle.load(handle)
    
x, H = solver.optimize(graph)
#%%
for node in graph.pose_nodes.values():
    M=node.M
    mu=node.mu
    plt.plot(M[0,3], M[1,3], "o")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[2]), 0.1*sin(mu[2]))
    
for node in graph.feature_nodes.values():
    M=node.M
    mu=node.mu
    plt.plot(M[0,3], M[1,3], "x")
    plt.arrow(M[0,3], M[1,3], 0.1*cos(mu[3]), 0.1*sin(mu[3]))
