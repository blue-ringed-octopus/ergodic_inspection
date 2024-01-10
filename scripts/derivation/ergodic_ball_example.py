# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:47:40 2024

@author: hibad
"""
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

class Bucket:
    def __init__(self,n, n_red):
        self.n=n
        self.n_red=n_red
        self.reset()
        
    def reset(self):
        pass 
          
    def sample(self):
        return np.random.randint(n)<self.n_red

    def detect(self):
        pass
buckets=[]
n=10
num_buckets=10
truth=[]
for i in range(num_buckets):
    n_red=np.random.randint(0,n)
    buckets.append(Bucket(n, n_red))
    truth.append((n_red/n)>0.5)
    
samples=[[] for i in range(num_buckets)]

steps=10
print(truth)

#%% random
P_anomaly=np.ones(num_buckets)*0.5
P_normal=np.ones(num_buckets)*0.5
samples=[[] for i in range(num_buckets)]

for i in range(steps):    
    bucket=np.random.randint(0,n)
    sample=buckets[bucket].sample()
    samples[bucket].append(sample)
    if sample:
        P_anomaly[bucket]*=0.75
        P_normal[bucket]*=0.25
    else:
        P_anomaly[bucket]*=0.25
        P_normal[bucket]*=0.75
        
    px=P_anomaly[bucket]+P_normal[bucket]
    P_anomaly[bucket]/=px
    P_normal[bucket]/=px
    
#%% ergodic
P_anomaly=np.ones(num_buckets)*0.5
P_normal=np.ones(num_buckets)*0.5
h=np.ones(num_buckets)*bernoulli.entropy(0.5)
samples=[[] for i in range(num_buckets)]

for i in range(steps):    
    h=h/sum(h)
    bucket=np.random.choice(range(num_buckets),p=h)
    sample=buckets[bucket].sample()
    samples[bucket].append(sample)
    if sample:
        P_anomaly[bucket]*=0.75
        P_normal[bucket]*=0.25
    else:
        P_anomaly[bucket]*=0.25
        P_normal[bucket]*=0.75
        
    px=P_anomaly[bucket]+P_normal[bucket]
    P_anomaly[bucket]/=px
    P_normal[bucket]/=px
    h[bucket]=bernoulli.entropy(P_anomaly[bucket])
print(P_anomaly)    
#%% max entropy
P_anomaly=np.ones(num_buckets)*0.5
P_normal=np.ones(num_buckets)*0.5
h=np.ones(num_buckets)*bernoulli.entropy(0.5)
samples=[[] for i in range(num_buckets)]

for i in range(steps):    
    bucket=np.argmax(h)
    sample=buckets[bucket].sample()
    samples[bucket].append(sample)
    if sample:
        P_anomaly[bucket]*=0.75
        P_normal[bucket]*=0.25
    else:
        P_anomaly[bucket]*=0.25
        P_normal[bucket]*=0.75
        
    px=P_anomaly[bucket]+P_normal[bucket]
    P_anomaly[bucket]/=px
    P_normal[bucket]/=px
    h[bucket]=bernoulli.entropy(P_anomaly[bucket])
    
print(P_anomaly)
