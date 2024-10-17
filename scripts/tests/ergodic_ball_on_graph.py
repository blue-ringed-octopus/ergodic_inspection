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
    w = weight.copy()
    w += np.ones(len(w))*0.0001
    w=w/sum(w)
    
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

class Region:
    def __init__(self, _id ,p_red):
        self.reset(p_red)
        self.neighbor=[]
        self.id = _id
        
    def reset(self, p_red):
        self.p_red=p_red 
          
    def sample(self, num):
        return np.random.rand(num)<self.p_red

    def detect(self):
        pass
    
    def add_neighbor(self, edges):
        edges=np.array(edges)
        self.neighbor = edges[edges[:,0]==self.id][:,1]

class Graph:
    def __init__(self):
        self.n = 7
        self.regions=[]

            
        self.edges=[[0, 0],
                     [0, 3],
                     [0, 1],
                     [1, 1],
                     [1, 2],
                     [1, 0],
                     [2, 2],
                     [2, 3],
                     [2, 1],
                     [3, 3],
                     [3, 0],
                     [3, 5],
                     [3, 2],
                     [4, 4],
                     [4, 5],
                     [4, 1],
                     [5, 5],
                     [5, 3],
                     [5, 6],
                     [5, 4],
                     [6, 6],
                     [6, 5]]
        
        for i in range(self.n):
            self.regions.append(Region(i, 0.5))
            self.regions[i].add_neighbor(self.edges)
            
    def reset(self,p_reds):
        for p_red, region in zip(p_reds, self.regions):
            region.reset(p_red)

def detect(samples,P_normal, P_anomaly, region, cutoff):
    for sample in samples:
        if sample:
            P_anomaly[region]*=(1+cutoff)/2
            P_normal[region]*=(cutoff)/2
        else:
            P_anomaly[region]*=(1-cutoff)/2
            P_normal[region]*=(2-cutoff)/2   
        
    px=P_anomaly[region]+P_normal[region]
    P_anomaly[region]/=px
    P_normal[region]/=px
    return  P_normal,  P_anomaly      

# params     
num_trial = 300
cutoff=0.2
graph= Graph()
p_reds=np.random.rand(graph.n)
graph.reset(p_reds)
truth=(p_reds)>cutoff
num_sample = 3
#%% random
print("random")
error_random=[[] for _ in range(num_trial)]
steps=30
for i in range(num_trial):
    p_reds=np.random.rand(graph.n)*0.5
    graph.reset(p_reds)
    truth=(p_reds)>cutoff
    
    P_anomaly=np.ones(graph.n)*0.5
    P_normal=np.ones(graph.n)*0.5
    region=0
    for _ in range(steps):    
        region=np.random.choice(graph.regions[region].neighbor)
        sample=graph.regions[region].sample(num_sample)

        P_normal,  P_anomaly = detect(sample,P_normal, P_anomaly, region, cutoff)  

    # error_random.append(max(abs(P_anomaly-truth)))
        error_random[i].append(np.linalg.norm(P_anomaly-truth))

#%% max entropy
error_max_entropy=[[] for _ in range(num_trial)]
print("max_ent")

for i in range(num_trial):
    p_reds=np.random.rand(graph.n)*0.5
    graph.reset(p_reds)
    truth=(p_reds)>cutoff
    
    P_anomaly=np.ones(graph.n)*0.5
    P_normal=np.ones(graph.n)*0.5
    h=np.ones(graph.n)*bernoulli.entropy(0.5)
    region=0
    for _ in range(steps):
        neighbors = graph.regions[region].neighbor
        region=neighbors[np.argmax(h[neighbors])]
        sample=graph.regions[region].sample(num_sample)
        
        P_normal,  P_anomaly = detect(sample,P_normal, P_anomaly, region, cutoff)  

        h[region]=bernoulli.entropy(P_anomaly[region])
        
    # error_max_entropy.append(max(abs(P_anomaly-truth)))
        error_max_entropy[i].append(np.linalg.norm(P_anomaly-truth))

#%% dicounted ergodic 
print("erg")

error_demc=[[] for _ in range(num_trial)]
for i in range(num_trial):
    print(i)
    p_reds=np.random.rand(graph.n)*0.5
    graph.reset(p_reds)
    truth=(p_reds)>cutoff
    
    P_anomaly=np.ones(graph.n)*0.5
    P_normal=np.ones(graph.n)*0.5
    h=np.ones(graph.n)*bernoulli.entropy(0.5)
    region=0
    for _ in range(steps):    
        P = DEMC(h, graph.edges)
        region=np.random.choice(range(graph.n),p=P[:,region])
        sample=graph.regions[region].sample(num_sample)
        P_normal,  P_anomaly = detect(sample,P_normal, P_anomaly, region, cutoff)  
           
        h[region]=bernoulli.entropy(P_anomaly[region])
        error_demc[i].append(np.linalg.norm(P_anomaly-truth))
#%% 
error_uniform=[[] for _ in range(num_trial)]
for i in range(num_trial):
    print(i)
    p_reds=np.random.rand(graph.n)*0.5
    graph.reset(p_reds)
    truth=(p_reds)>cutoff
    
    P_anomaly=np.ones(graph.n)*0.5
    P_normal=np.ones(graph.n)*0.5
    h=np.ones(graph.n)*bernoulli.entropy(0.5)
    region=0
    P = DEMC(h, graph.edges)

    for _ in range(steps):    
        region=np.random.choice(range(graph.n),p=P[:,region])
        sample=graph.regions[region].sample(num_sample)
        P_normal,  P_anomaly = detect(sample,P_normal, P_anomaly, region, cutoff)  
           
        error_uniform[i].append(np.linalg.norm(P_anomaly-truth))
        
#%%
error_tenstep = [[] for _ in range(num_trial)]
for i in range(num_trial):
    print(i)
    p_reds=np.random.rand(graph.n)*0.5
    graph.reset(p_reds)
    truth=(p_reds)>cutoff
    
    P_anomaly=np.ones(graph.n)*0.5
    P_normal=np.ones(graph.n)*0.5
    h=np.ones(graph.n)*bernoulli.entropy(0.5)
    region=0
    for j in range(steps):  
        if not j%10:
            P = DEMC(h, graph.edges)
        region=np.random.choice(range(graph.n),p=P[:,region])
        sample=graph.regions[region].sample(num_sample)
        P_normal,  P_anomaly = detect(sample,P_normal, P_anomaly, region, cutoff)  
        h[region]=bernoulli.entropy(P_anomaly[region])
   
        error_tenstep[i].append(np.linalg.norm(P_anomaly-truth))        
#%%
error_random = np.array(error_random)
error_max_entropy = np.array(error_max_entropy)
error_demc = np.array(error_demc)

def box_plot(dat, pos):
    bp = plt.boxplot(dat, positions=pos, meanline=True, showfliers=False, patch_artist=True)
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
# print("random", str(np.mean(error_random))+"+-"+ str(np.std(error_random)))

# print("max entr.", str(np.mean(error_max_entropy))+"+-"+ str(np.std(error_max_entropy)))

# print("ergodic", str(np.mean(error_demc))+"+-"+ str(np.std(error_demc)))
plt.plot(range(steps),np.mean(error_random, 0), "--", color = "blue")
plt.plot(range(steps),np.mean(error_max_entropy, 0), "--", color = "green")
plt.plot(range(steps),np.mean(error_demc, 0), "--", color = "red")
plt.plot(range(steps),np.mean(error_uniform, 0), "--", color = "pink")
plt.plot(range(steps),np.mean(error_tenstep, 0), "--", color = "black")

# for i in range(num_trial):
#     plt.plot(range(steps),error_demc[i,:], "--", color = "red", alpha=0.1)
#     plt.plot(range(steps),error_max_entropy[i,:], "--", color = "green", alpha=0.1)
#     plt.plot(range(steps),error_random[i,:], "--", color = "blue", alpha=0.1)


# for i in  range(steps):  
#     box_plot(error_random[:,i], [i])
#     box_plot(error_max_entropy[:,i], [i])
#     box_plot(error_demc[:,i], [i])
   
