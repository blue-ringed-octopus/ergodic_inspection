# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:47:40 2024

@author: hibad
"""
import numpy as np
import cvxpy as cp
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
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

def UBFMMC(weight, edges, transform=True):
    n=len(weight)
    P= cp.Variable((n,n))


    sigma=weight
    q=np.sqrt(sigma)
    Q=np.diag(q)
    constrains=[P@sigma==sigma,
                cp.min(P)>=0,
                np.ones(n)@P==np.ones(n)]
    
    for i in range(n):
        for j in range(n):
            if ([i,j] not in edges):
                constrains.append(P[j,i]==0)

    if transform:
        prob = cp.Problem(cp.Minimize(cp.norm(np.linalg.inv(Q)@P@(Q)-outer(q,q),2)),
                            constrains)

    else:
          prob = cp.Problem(cp.Minimize(cp.norm(P-outer(sigma,np.ones(n)),2)),
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

def DEMC(weight, edges):
    w=weight/sum(weight)
    n = len(w)
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

class Region:
    def __init__(self, _id ,p_red):
        self.reset(p_red)
        self.neighbor=[]
        self.id = _id
    def reset(self, p_red):
        self.p_red=p_red 
          
    def sample(self):
        return np.random.rand()<self.p_red

    
    def detect(self):
        pass
    
    def add_neighbor(self, edges):
        edges=np.array(edges)
        self.neighbor = edges[edges[:,0]==self.id][:,1]

class Graph:
    def __init__(self, num_regions):
        self.regions=[]
        self.edges=[[0,0],
                    [1,1],
                    [2,2],
                    [3,3],
                    [4,4],
                    [5,5],
                    [6,6],
                     
                     [0,1],
                     [0,3],
                     [1,0], 
                     [1,2],
                     [1,4],
                     [2,1],
                     [2,3],
                     [3,0],
                     [3,2],
                     [3,5],
                    #[4,1],
                     [4,5],
                     [5,3],
                     [5,4],
                     [5,6],
                     [6,5]]
# params     
num_trial = 100
cutoff=0.2
num_regions=7
graph= Graph(num_regions)

truth=[]
for i in range(num_regions):
    p_red=np.random.rand()*0.5
    regions.append(Region(i, p_red))
    truth.append((p_red)>cutoff)
    regions[i].add_neighbor(edges)


samples=[[] for i in range(num_regions)]
print(truth)


steps_range=np.arange(1,10)
#%% random
error_random=[]
steps=10
for _ in range(num_trial):

    P_anomaly=np.ones(num_regions)*0.5
    P_normal=np.ones(num_regions)*0.5
    samples=[[] for i in range(num_regions)]
    region=0
    for i in range(steps):    
        region=np.random.choice(regions[region].neighbor)
        sample=regions[region].sample()
        samples[region].append(sample)
        if sample:
            P_anomaly[region]*=(1+cutoff)/2
            P_normal[region]*=(cutoff)/2
        else:
            P_anomaly[region]*=(1-cutoff)/2
            P_normal[region]*=(2-cutoff)/2
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        
    # error_random.append(max(abs(P_anomaly-truth)))
    error_random.append(np.linalg.norm(P_anomaly-truth))

#%% max entropy
error_max_entropy=[]

for _ in range(num_trial):

    P_anomaly=np.ones(num_regions)*0.5
    P_normal=np.ones(num_regions)*0.5
    h=np.ones(num_regions)*bernoulli.entropy(0.5)
    samples=[[] for i in range(num_regions)]
    region=0
    for i in range(steps):
        neighbors = regions[region].neighbor
        region=neighbors[np.argmax(h[neighbors])]
        sample=regions[region].sample()
        samples[region].append(sample)
        if sample:
            P_anomaly[region]*=(1+cutoff)/2
            P_normal[region]*=(cutoff)/2
        else:
            P_anomaly[region]*=(1-cutoff)/2
            P_normal[region]*=(2-cutoff)/2
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        h[region]=bernoulli.entropy(P_anomaly[region])
        
    # error_max_entropy.append(max(abs(P_anomaly-truth)))
    error_max_entropy.append(np.linalg.norm(P_anomaly-truth))
#%% fastest mixing 
error_fmmc=[]
for _ in range(num_trial):
    P_anomaly=np.ones(num_regions)*0.5
    P_normal=np.ones(num_regions)*0.5
    h=np.ones(num_regions)*bernoulli.entropy(0.5)
    samples=[[] for i in range(num_regions)]
    region=0
    for i in range(steps):    
        h=h/sum(h)
        P = UBFMMC(h, edges)
        region=np.random.choice(range(num_regions),p=P[:,region])
        sample=regions[region].sample()
        samples[region].append(sample)
        if sample:
            P_anomaly[region]*=(1+cutoff)/2
            P_normal[region]*=(cutoff)/2
        else:
            P_anomaly[region]*=(1-cutoff)/2
            P_normal[region]*=(2-cutoff)/2
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        h[region]=bernoulli.entropy(P_anomaly[region])
    # error_fmmc.append(max(abs(P_anomaly-truth)))
    error_fmmc.append(np.linalg.norm(P_anomaly-truth))
#%% dicounted ergodic 
error_demc=[]
for _ in range(num_trial):
    P_anomaly=np.ones(num_regions)*0.5
    P_normal=np.ones(num_regions)*0.5
    h=np.ones(num_regions)*bernoulli.entropy(0.5)
    samples=[[] for i in range(num_regions)]
    region=0
    for i in range(steps):    
        h=h/sum(h)
        P = DEMC(h, edges)
        region=np.random.choice(range(num_regions),p=P[:,region])
        sample=regions[region].sample()
        samples[region].append(sample)
        if sample:
            P_anomaly[region]*=(1+cutoff)/2
            P_normal[region]*=(cutoff)/2
        else:
            P_anomaly[region]*=(1-cutoff)/2
            P_normal[region]*=(2-cutoff)/2
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        h[region]=bernoulli.entropy(P_anomaly[region])
    # error_demc.append(max(abs(P_anomaly-truth)))
    error_demc.append(np.linalg.norm(P_anomaly-truth))

#%%
print("random", str(np.mean(error_random))+"+-"+ str(np.std(error_random)))

print("max entr.", str(np.mean(error_max_entropy))+"+-"+ str(np.std(error_max_entropy)))

print("ergodic", str(np.mean(error_fmmc))+"+-"+ str(np.std(error_fmmc)))

print("ergodic2", str(np.mean(error_demc))+"+-"+ str(np.std(error_demc)))

