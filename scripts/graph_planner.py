# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:52:53 2024

@author: hibad
"""
import numpy as np
import cvxpy as cp
import cv2
np.set_printoptions(precision=2)


def outer(a, b):
    a = cp.Expression.cast_to_const(a)  # if a is an Expression, return it unchanged.
    assert a.ndim == 1
    b = cp.Expression.cast_to_const(b)
    assert b.ndim == 1
    a = cp.reshape(a, (a.size, 1))
    b = cp.reshape(b, (1, b.size))
    expr = a @ b
    return expr

class Graph_Planner:
    def __init__(self, nodes, edges, strategy = "ergodic"):
        self.strategy = strategy 
        self.num_regions = len(nodes)
        self.edges = edges
        self.adjacency = np.zeros((self.num_regions,self.num_regions))
        for i, j in self.edges:
            self.adjacency[j,i]=1
        

    def discounted_ergodic_markov_chain(self, weight):
        
        w=weight/sum(weight)
        print("w1: ", w)

        w = w + 0.001
        w = w/sum(w)
        print("w2: ", w)

        n = self.num_regions
        edges = self.edges
        P= cp.Variable((n,n))
        
        
        q = np.sqrt(w)
        Q=np.diag(q)

        constrains=[
                    P@w==w,
                    cp.min(P)>=0,
                    np.ones(n)@P==np.ones(n)
                    ]
        
        for i in range(n):
            for j in range(n):
                if ([i,j] not in edges):
                    constrains.append(P[j,i]==0)
        # P_tilde = P - 2*outer(w,np.ones(n))
        P_tilde = np.linalg.inv(Q)@P@(Q)-2*outer(q,q)
        

        objective = cp.lambda_max(1/2*(P_tilde+P_tilde.T))

        prob = cp.Problem(cp.Minimize(objective),
                          constrains)
    
        prob.solve()
        P=P.value
        
        P[P<0]=0
        for i in range(n):
            for j in range(n):
                if ([i,j] not in edges):
                    P[j,i]=0
        P=P/sum(P, 0) 
        return P
   
    def FMMC(self, weight, transform=True):
        w =weight/sum(weight)
        n = self.num_regions
        P= cp.Variable((n,n))


        q=np.sqrt(w)
        Q=np.diag(q)
        constrains=[
                   # P@w==w,
                    P@np.diag(w) == np.diag(w)@P.T,
                    cp.min(P)>=0,
                    np.ones(n)@P==np.ones(n)]
        
        for i in range(n):
            for j in range(n):
                if ([i,j] not in self.edges):
                    constrains.append(P[j,i]==0)

        if transform:
            prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-outer(q,q),2)),
                                constrains)

        else:
              prob = cp.Problem(cp.Minimize(cp.norm(P-outer(w,np.ones(n)),2)),
                                constrains)
     

        prob.solve()
        P=P.value

        P[P<0]=0
        for i in range(n):
            for j in range(n):
                if ([i,j] not in self.edges):
                    P[j,i]=0
        P=P/sum(P, 0)

        return P
    def set_weights(self, weight):
        self.w = weight.copy()
        if self.strategy == "ergodic":
            P = self.discounted_ergodic_markov_chain(weight)
            
        elif self.strategy == "random":
            P = self.adjacency.copy()
            P=P/sum(P, 0)
            
        elif self.strategy == "greedy":
            print(weight/np.sum(weight))
            P =  self.adjacency.copy()
            for i in range(self.num_regions):
                P[:, i] = P[:, i] * weight 
                P[:, i]= P[:, i]==np.max(P[:, i])
        self.P= P
        return P.copy()
    
    def get_next_region(self, current_region):
        # if self.strategy == "ergodic":
        #     P = self.discounted_ergodic_markov_chain(weight)
        #     region = np.random.choice(range(self.num_regions),p=P[:,current_region])
            
        # elif self.strategy == "random":
        #     P = self.adjacency.copy()
        #     P=P/sum(P, 0)
        #     region = np.random.choice(range(self.num_regions),p=P[:,current_region])
            
        # elif self.strategy == "greedy":
        #     P =  self.adjacency.copy()[:,current_region]
        #     P = P * weight 
        #     region = np.argmax(P)
        P = self.P.copy()
        region = np.random.choice(range(self.num_regions),p=P[:,current_region])
        return region, P
    



if __name__ == '__main__':
    from map_manager import Map_Manager   
    import matplotlib.pyplot as plt 
    
    manager = Map_Manager("../resources/sim/")
    nodes, edges, _ = manager.get_graph(1)     
    # edges.remove([1,4])
    nodes = [0,1,2]
    edges = [[0,0], [0,1], [1,0], [1,1], [1,2],[2,1] ,[2,2], [0,2]]
    n = len(nodes)
    K=10
    planner = Graph_Planner(nodes, edges)
    num_trial = 100
    error_demc = [[] for _ in range(num_trial)]
    error_fmmc = [[] for _ in range(num_trial)]
    
    for trial in range(num_trial):
        w = np.random.rand(n)
        w = w/sum(w)
        x0 = np.zeros(n)
        x0[np.random.randint(0,n)] = 1
        
        P_demc = planner.discounted_ergodic_markov_chain(w)
        x = x0.copy()
        x_demc=[x.copy()]
        x_hat_demc = [x.copy()]
        error_demc[trial].append(np.linalg.norm(x-w))
        step_error_demc =  [np.linalg.norm(x_hat_demc[0]-w)]
        
        for i in range(K):
            x = P_demc@x
            x_demc.append(x.copy())
            x_hat_demc.append(np.sum(x_demc,0)/len(x_demc))
            error_demc[trial].append(np.linalg.norm(x_hat_demc[i+1]-w))
            step_error_demc.append(np.linalg.norm(x-w))
            
        x = x0.copy()
        x_fmmc=[x]
        x_hat_fmmc = [x]
        P_fmmc = planner.FMMC(w)

        error_fmmc[trial].append(np.linalg.norm(x_hat_fmmc[0]-w))
        step_error_fmmc =  [np.linalg.norm(x_hat_fmmc[0]-w)]
    
        for i in range(K):
            x_fmmc.append(P_fmmc@x_fmmc[i].copy())
            x_hat_fmmc.append(np.sum(x_fmmc,0)/len(x_fmmc))
            error_fmmc[trial].append(np.linalg.norm(x_hat_fmmc[i+1]-w))
            step_error_fmmc.append(np.linalg.norm(x_fmmc[i+1]-w))

    error_demc = np.array(error_demc)
    error_fmmc = np.array(error_fmmc)

    plt.figure(dpi = 1200)
    # plt.figure()
    bp = plt.boxplot(error_demc, positions=range(K+1), meanline=True, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor([0.2,0.2,1,0.5])
        patch.set(edgecolor =[0.2,0.2,0.5, 0.5],
                linewidth = 1.5,
                )
    for whisker in bp['whiskers']:
        whisker.set(color =[0.2,0.2,0.5, 0.5],
                linewidth = 1.5,
                )
    for cap in bp['caps']:
        cap.set(color =[0.2,0.2,0.5, 0.5],
                linewidth = 1.5)
        
    for median in bp['medians']:
        median.set(color =[0.2,0.2,0.5, 0.5],
                   linewidth = 1.5)

    bp = plt.boxplot(error_fmmc, positions=range(K+1), meanline=True, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor([1,0.2,0.2,0.5])
        patch.set(edgecolor =[0.5,0.2,0.2, 0.5],
                linewidth = 1.5,
                )
    for whisker in bp['whiskers']:
        whisker.set(color =[0.5,0.2,0.2, 0.5],
                linewidth = 1.5,
                )
    for cap in bp['caps']:
        cap.set(color =[0.5,0.2,0.2, 0.5],
                linewidth = 1.5)
        
    for median in bp['medians']:
        median.set(color =[0.5,0.2,0.2, 0.5],
                   linewidth = 1.5)
        
    plt.plot(np.mean(error_fmmc,0),"r--", label="FMMC")
    plt.plot(np.mean(error_demc,0),"b--", label="Optimal Ergodic")

    plt.legend()
    plt.title("Ergodicity Error ("+str(num_trial)+" trials)")
    plt.xlabel("k")
    plt.ylabel(r'$\|\hat{r}_k-\bar{r}\|$')
   
    plt.figure(dpi = 1200)
    pairwise_error = error_fmmc -error_demc
    plt.plot([0, K], [0,0], "--", color="grey", linewidth="1")
    plt.plot(np.mean(pairwise_error,0),"k--", label=" FMMC - Optimal Ergodic")
    bp = plt.boxplot(pairwise_error, positions=range(K+1), meanline=True, showfliers=False)
    plt.title("Pairwise Difference")
    plt.xlabel("k")
    plt.ylabel(r'$err_{FMMC} - err_{optimal}$')
    # plt.figure()
    # plt.plot(step_error_demc, label="Discounted Ergodic")
    # plt.plot(step_error_fmmc, label="FMMC")
    # plt.legend()
    # plt.title("Stepwise Error")
    # plt.xlabel("k")
    # plt.ylabel(r'$\|\rho_k-\bar{r}\|$')
