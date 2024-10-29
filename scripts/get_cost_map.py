# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:01:55 2022

@author: hibad
"""
import math
import sys
import open3d as o3d
import colorsys as cs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from numba import cuda
import pickle

sim = False
if sim:
    mesh_path = "../resources/sim/Ballast.STL"
    output_path = '../resources/sim/costmap.pickle'
else:
    mesh_path = "../resources/real/Ballast.STL"
    output_path = '../resources/real/costmap.pickle'
resolution=0.05
kernel_size=(5,5)

if sim:
    robot_radius=0.175/2/resolution
    inflation_radius= 0.5/resolution
    cost_scaling_factor = 2* resolution
else:
    robot_radius=0.25/2/resolution
    inflation_radius= 0.25/resolution
    cost_scaling_factor = 1.5* resolution
#%% Import FOD clouds
mesh = o3d.io.read_triangle_mesh(mesh_path)
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
# o3d.visualization.draw_geometries([frame, mesh])
box = mesh.get_axis_aligned_bounding_box()
min_bound = np.array([box.min_bound[0],box.min_bound[1], 0.05 ])
max_bound = np.array([box.max_bound[0],box.max_bound[1], 0.2 ])
# min_bound = np.array([box.min_bound[0],box.min_bound[1], 0.1 ])
# max_bound = np.array([box.max_bound[0],box.max_bound[1], 0.2 ])
box.min_bound = min_bound
box.max_bound = max_bound
pc = mesh.sample_points_uniformly(
    number_of_points=1000000)
pc=pc.crop(box)

pt = np.asarray(pc.points)
pt[:,2]=0
pc.points=o3d.utility.Vector3dVector(pt)


voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=resolution)
origin = voxel_grid.get_voxel([0,0,0])
voxels = voxel_grid.get_voxels()  # returns list of voxels
indices = np.stack(list(vx.grid_index for vx in voxels))[:,0:2]
matrix=np.zeros(np.max(indices, axis=0)+1)
matrix[indices[:,0],indices[:,1]]=255
_, matrix = cv2.threshold(matrix,127,255,cv2.THRESH_BINARY)

masked_map = np.uint8(matrix.copy())

x_shape, y_shape = masked_map.shape[:2]
mask = np.zeros((x_shape + 2, y_shape + 2), np.uint8)
seed = [10, 50]
cv2.floodFill(masked_map, mask, (np.uint32(seed[0]), np.uint32(seed[1])), 255)
masked_map=255-masked_map+matrix
plt.imshow(masked_map.T, origin="lower")    
#%% map to cost map
TPB=32
@cuda.jit
def inflation_kernel(d_out, d_image, radius,cost_scaling_factor,robot_radius):
    nx=d_image.shape[0]
    ny=d_image.shape[1]
    min_dist=radius
    x,y=cuda.grid(2)
    if x<nx and y<ny:
        d_out[x,y]=0
        for i in range(nx):
            for j in range(ny):
                dist=math.sqrt((x-i)**2+(y-j)**2)
                if d_image[i,j]==255:
                    if dist<min_dist:
                        min_dist=dist
                        if dist>robot_radius:
                            d_out[x,y]=255*math.exp((-1.0 * cost_scaling_factor * (dist-robot_radius)))
                        else:
                            d_out[x,y]=255
                     

        
def inflation(image, inflation_radius, cost_scaling_factor):
    result=deepcopy(image)
    w=image.shape[0]
    h=image.shape[1]
    for i in range(w):
        for j in range(h):
            yy, xx = np.meshgrid(np.linspace(0, h-1, h), np.linspace(0, w-1, w))
            mask = (xx-i)*2 + (yy-j)**2 < inflation_radius**2
            result[i,j]=np.max(image[mask])
    return result

def inflation_par(image, inflation_radius, cost_scaling_factor,robot_radius):
    nx=image.shape[0]
    ny=image.shape[1]
    d_image=cuda.to_device(image)
    d_out=cuda.device_array((nx,ny),dtype=np.float32)
    threads = TPB, TPB
    blocks=(nx+TPB-1)//TPB, (ny+TPB-1)//TPB
    inflation_kernel[blocks, threads](d_out,d_image, inflation_radius, cost_scaling_factor,robot_radius)
    return d_out.copy_to_host()


cost = inflation_par(masked_map, inflation_radius, cost_scaling_factor,robot_radius)
# cost=cost.T
occupancy = cost.copy()
occupancy[occupancy != 255] =0 
plt.figure()
plt.imshow(occupancy.T, origin="lower")
plt.title("occupancy")
plt.figure()
plt.imshow(cost.T, origin="lower")
plt.title("cost")

resolution = (max_bound-min_bound)[0:2]
resolution = resolution/[x_shape, y_shape]

with open(output_path, 'wb') as handle:
    pickle.dump({"occupancy_map": occupancy/255*100, "costmap": cost/255*100, "resolution": resolution,"origin":voxel_grid.origin, "bounds":{"min": min_bound, "max": max_bound} }, handle)
