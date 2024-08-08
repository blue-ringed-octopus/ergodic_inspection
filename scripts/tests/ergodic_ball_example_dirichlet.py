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
num_buckets=7
truth=[]
cutoff=0.1

for i in range(num_buckets):
    n_red=np.random.randint(0,n)
    buckets.append(Bucket(n, n_red))
    truth.append((n_red/n)>cutoff)
    
samples=[[] for i in range(num_buckets)]
print(truth)

num_trial = 1
steps_range=np.arange(1,10)
#%% random
error_random=[]
steps=1000
buckets[0].n_red = 3
for _ in range(num_trial):

    c_anomaly=np.zeros(num_buckets)+0.001
    c_normal=np.zeros(num_buckets) +0.001
    samples=[[] for i in range(num_buckets)]
    p_anomaly = []
    for i in range(steps):    
        # bucket=np.random.randint(0,n)
        bucket=0
        sample=buckets[bucket].sample()
        samples[bucket].append(sample)
        if sample:
            s = ((1+cutoff)/2+(cutoff)/2)
            c_anomaly[bucket] +=(1+cutoff)/2/s
            c_normal[bucket]+=(cutoff)/2/s
        else:
            s = ((1-cutoff)/2+(2-cutoff)/2)
            c_anomaly[bucket]+=(1-cutoff)/2/s
            c_normal[bucket]+=(2-cutoff)/2/s
            
        # c_anomaly[bucket]/=px
        # c_normal[bucket]/=px
        p_anomaly.append(c_anomaly/(c_anomaly+c_normal))
    error_random.append(max(abs(c_anomaly/(c_anomaly+c_normal)-truth)))
p_anomaly = np.array(p_anomaly)    
plt.plot(p_anomaly[:,0])
#%% ergodic
error_ergodic=[]
for _ in range(num_trial):

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
    error_ergodic.append(max(abs(P_anomaly-truth)))
#%% max entropy
error_max_entropy=[]

for _ in range(num_trial):

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
        
    error_max_entropy.append(max(abs(P_anomaly-truth)))

print(str(np.mean(error_max_entropy))+"+-"+ str(np.std(error_max_entropy)))

print(str(np.mean(error_ergodic))+"+-"+ str(np.std(error_ergodic)))
