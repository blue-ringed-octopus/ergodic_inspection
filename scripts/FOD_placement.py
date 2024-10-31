# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:23:54 2024

@author: hibad
"""
import matplotlib.pyplot as plt
import numpy as np
import yaml 
import pickle 
import cv2
is_sim = True
num_trial = 30
if is_sim:
    path = "../resources/"+"sim/"
else:
    path = "../resources/"+"real/"

with open(path+"fod_list.yaml") as stream:
    try:
        fod_candidates = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)  

with open(path+'costmap.pickle', 'rb') as stream:
    costmap = pickle.load(stream)  
occupancy_map =  costmap["occupancy_map"]
occupancy_map = occupancy_map.astype(np.uint8)
plt.imshow(occupancy_map, cmap="Grays")
edge = cv2.Canny(occupancy_map, 100, 200)
plt.imshow(edge, cmap="Grays")
idx = np.where(edge==255)
num_pix = len(idx[0])
scale = 20

for trial in range(num_trial):
    num_fod = np.random.randint(3,len(fod_candidates)+1)
    fods = np.random.choice(fod_candidates, num_fod, False)
    fod_map = occupancy_map.copy()
    fod_map = cv2.cvtColor(fod_map, cv2.COLOR_GRAY2BGR)
    h,w,c = fod_map.shape
    fod_map = cv2.resize(fod_map, (w*scale, h*scale))
    for fod in fods:
        i = np.random.choice(range(num_pix))
        x = idx[0][i]
        y = idx[1][i]
        fod_map = cv2.circle(fod_map, (y*scale,x*scale), 10, (255,0,0), thickness=-1)
        fod_map = cv2.putText(fod_map, fod, (y*scale,x*scale), cv2.FONT_HERSHEY_SIMPLEX , 2, (255,255,255) , 5)
    plt.figure(dpi = 500)
    plt.imshow(fod_map) 
    plt.title("trial: " + str(trial))   
    plt.savefig(path+"fod_loc/"+"fod_trial_"+str(trial))