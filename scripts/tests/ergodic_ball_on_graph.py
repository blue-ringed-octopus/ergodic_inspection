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

class Region:
    def __init__(self, _id ,n, n_red):
        self.n=n
        self.n_red=n_red
        self.reset()
        self.neighbor=[]
        self.id = _id
    def reset(self):
        pass 
          
    def sample(self):
        return np.random.randint(n)<self.n_red

    
    def detect(self):
        pass
    
    def add_neighbor(self, edges):
        edges=np.array(edges)
        self.neighbor = edges[edges[:,0]==self.id][:,1]

# params     
num_trial = 1000
cutoff=0.5
regions=[]
n=10
num_regions=7
edges=[[0,0],
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
       [4,1],
       [4,5],
       [5,3],
       [5,4],
       [5,6],
       [6,5]]

truth=[]
for i in range(num_regions):
    n_red=np.random.randint(0,n)
    regions.append(Region(i, n, n_red))
    truth.append((n_red/n)>0.5)
    regions[i].add_neighbor(edges)


samples=[[] for i in range(num_regions)]
print(truth)


steps_range=np.arange(1,10)
#%% random
error_random=[]
steps=100
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
            P_anomaly[region]*=0.75
            P_normal[region]*=0.25
        else:
            P_anomaly[region]*=0.25
            P_normal[region]*=0.75
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        
    error_random.append(max(abs(P_anomaly-truth)))

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
            P_anomaly[region]*=0.75
            P_normal[region]*=0.25
        else:
            P_anomaly[region]*=0.25
            P_normal[region]*=0.75
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        h[region]=bernoulli.entropy(P_anomaly[region])
        
    error_max_entropy.append(max(abs(P_anomaly-truth)))
    
#%% ergodic
error_ergodic=[]
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
            P_anomaly[region]*=0.75
            P_normal[region]*=0.25
        else:
            P_anomaly[region]*=0.25
            P_normal[region]*=0.75
            
        px=P_anomaly[region]+P_normal[region]
        P_anomaly[region]/=px
        P_normal[region]/=px
        h[region]=bernoulli.entropy(P_anomaly[region])
    error_ergodic.append(max(abs(P_anomaly-truth)))


#%%
print("random", str(np.mean(error_random))+"+-"+ str(np.std(error_random)))

print("max entr.", str(np.mean(error_max_entropy))+"+-"+ str(np.std(error_max_entropy)))

print("ergodic", str(np.mean(error_ergodic))+"+-"+ str(np.std(error_ergodic)))
