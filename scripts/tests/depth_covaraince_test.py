# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:20:20 2024

@author: hibad
"""
from copy import deepcopy
from scipy.stats import chi2 
from scipy.spatial import KDTree
import pyrealsense2 as rs
import numpy as np
from numpy.linalg import inv, det
import cv2
import open3d as o3d
from pupil_apriltags import Detector
import os
os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")
from numba import cuda
import time
def get_rs_param(cfg):
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    return [intr.fx, intr.fy, intr.ppx, intr.ppy]

TPB=32
@cuda.jit()
def cloud_cov_kernel(d_out, d_depth, d_Q, d_T):
    i,j = cuda.grid(2)
    nx, ny=d_depth.shape

    if i<nx and j<ny:
        n=int(j+i*ny)
        d = d_depth[i,j]
        if not d == 0:
            J00 = d*d_T[0,0]
            J01 = d*d_T[0,1]
            J02 = d_T[0,2] + i*d_T[0,0] + j*d_T[0,1]
            J10 = d*d_T[1,0]
            J11 = d*d_T[1,1]
            J12 = d_T[1,2] + i*d_T[1,0] + j*d_T[1,1]
            J20 = d*d_T[2,0]
            J21 =  d*d_T[2,1]
            J22 = d_T[2,2] + i*d_T[2,0] + j*d_T[2,1]
        
            d_out[n,0,0] = d_Q[0,0]*J00**2 + d_Q[1,1]*J01**2 + d_Q[2,2]*J02**2
            d_out[n,0,1] = d_Q[0,0]*J00*J10 + d_Q[1,1]*J01*J11 + d_Q[2,2]*J02*J12
            d_out[n,0,2] = d_Q[0,0]*J00*J20 + d_Q[1,1]*J01*J21 + d_Q[2,2]*J02*J22
            
            d_out[n,1,0] = d_out[n,0,1] 
            d_out[n,1,1] = d_Q[0,0]*J10**2 + d_Q[1,1]*J11**2 + d_Q[2,2]*J12**2
            d_out[n,1,2] = d_Q[0,0]*J10*J20 + d_Q[1,1]*J11*J21 + d_Q[2,2]*J12*J22
            
            d_out[n,2,0] = d_out[n,0,2]
            d_out[n,2,1] = d_out[n,1,2]
            d_out[n,2,2] = d_Q[0,0]*J20**2 + d_Q[1,1]*J21**2 + d_Q[2,2]*J22**2
            
def get_cloud_covariance_par(depth, Q, T):
    nx, ny=depth.shape
    d_depth=cuda.to_device(depth)
    d_Q=cuda.to_device(Q)
    d_T=cuda.to_device(T)
    d_out=cuda.device_array((nx*ny, 3, 3),dtype=(np.float64))
    thread=(TPB, TPB)
    blocks=((nx+TPB-1)//TPB,(ny+TPB-1)//TPB)
    cloud_cov_kernel[blocks, thread](d_out, d_depth,d_Q, d_T)
    cov=d_out.copy_to_host()
    return cov

@cuda.jit()
def md_kernel(d_out, d_epsilon, d_cov, d_normal, d_p, d_mu):
    i = cuda.grid(1)
    n=d_out.shape[0]
    if i<n:
        nmu=d_mu[i,0]*d_normal[i,0]+d_mu[i,1]*d_normal[i,1]+d_mu[i,2]*d_normal[i,2]
        npoint=d_p[i,0]*d_normal[i,0]+d_p[i,1]*d_normal[i,1]+d_p[i,2]*d_normal[i,2]
        ncn=d_cov[i,0,0]*d_normal[i,0]**2 + 2*d_cov[i,0,1]*d_normal[i,0]*d_normal[i,1] + 2*d_cov[i,0,2]*d_normal[i,0]*d_normal[i,2] + d_cov[i,1,1]*d_normal[i,1]**2 + 2*d_cov[i,1,2]*d_normal[i,1]*d_normal[i,2] + d_cov[i,2,2]*d_normal[i,2]**2

        if npoint<=nmu:
            d0 = 0 
        else:
            d0 = (nmu-npoint)**2/ncn

        if npoint>=nmu+d_epsilon:
            d1 = 0
        else:
            d1 = (thres + nmu-npoint)**2/ncn

        d_out[i,0] = d0
        d_out[i,1] = d1

def get_md_par(points, mu, epsilon, cov, normal):
   n = points.shape[0]
   d_mu=cuda.to_device(mu)
   d_cov = cuda.to_device(cov)
   d_normal = cuda.to_device(normal)
   d_p = cuda.to_device(points)
   thread=TPB
   d_out=cuda.device_array((n,2),dtype=(np.float64))
   blocks=(n+TPB-1)//TPB
   md_kernel[blocks, thread](d_out, epsilon, d_cov, d_normal, d_p, d_mu)
   return d_out.copy_to_host()

@cuda.jit()
def global_cov_kernel(d_out, d_point_cov, d_T, d_T_cov):
    i = cuda.grid(1)
    n=d_out.shape[0]
    if i<n:
        d_out[i,0,0] = d_point_cov[i,0,0]*d_T[0,0]**2 + 2*d_point_cov[i,0,1]*d_T[0,0]*d_T[0,1] + 2*d_point_cov[i,0,2]*d_T[0,0]*d_T[0,2] + d_point_cov[i,1,1]*d_T[0,1]**2 + 2*d_point_cov[i,1,2]*d_T[0,1]*d_T[0,2] + d_point_cov[i,2,2]*d_T[0,2]**2
        d_out[i,0,1] = d_T[1,0]*(d_point_cov[i,0,0]*d_T[0,0] + d_point_cov[i,0,1]*d_T[0,1] + d_point_cov[i,0,2]*d_T[0,2]) + d_T[1,1]*(d_point_cov[i,0,1]*d_T[0,0] + d_point_cov[i,1,1]*d_T[0,1] + d_point_cov[i,1,2]*d_T[0,2]) + d_T[1,2]*(d_point_cov[i,0,2]*d_T[0,0] + d_point_cov[i,1,2]*d_T[0,1] + d_point_cov[i,2,2]*d_T[0,2])
        d_out[i,0,2] = d_T[2,0]*(d_point_cov[i,0,0]*d_T[0,0] + d_point_cov[i,0,1]*d_T[0,1] + d_point_cov[i,0,2]*d_T[0,2]) + d_T[2,1]*(d_point_cov[i,0,1]*d_T[0,0] + d_point_cov[i,1,1]*d_T[0,1] + d_point_cov[i,1,2]*d_T[0,2]) + d_T[2,2]*(d_point_cov[i,0,2]*d_T[0,0] + d_point_cov[i,1,2]*d_T[0,1] + d_point_cov[i,2,2]*d_T[0,2])


        d_out[i,1,1] = d_point_cov[i,0,0]*d_T[1,0]**2 + 2*d_point_cov[i,0,1]*d_T[1,0]*d_T[1,1] + 2*d_point_cov[i,0,2]*d_T[1,0]*d_T[1,2] + d_point_cov[i,1,1]*d_T[1,1]**2 + 2*d_point_cov[i,1,2]*d_T[1,1]*d_T[1,2] + d_point_cov[i,2,2]*d_T[1,2]**2
        d_out[i,1,2] = d_T[2,0]*(d_point_cov[i,0,0]*d_T[1,0] + d_point_cov[i,0,1]*d_T[1,1] + d_point_cov[i,0,2]*d_T[1,2]) + d_T[2,1]*(d_point_cov[i,0,1]*d_T[1,0] + d_point_cov[i,1,1]*d_T[1,1] + d_point_cov[i,1,2]*d_T[1,2]) + d_T[2,2]*(d_point_cov[i,0,2]*d_T[1,0] + d_point_cov[i,1,2]*d_T[1,1] + d_point_cov[i,2,2]*d_T[1,2])

        d_out[i,2,2] = d_point_cov[i,0,0]*d_T[2,0]**2 + 2*d_point_cov[i,0,1]*d_T[2,0]*d_T[2,1] + 2*d_point_cov[i,0,2]*d_T[2,0]*d_T[2,2] + d_point_cov[i,1,1]*d_T[2,1]**2 + 2*d_point_cov[i,1,2]*d_T[2,1]*d_T[2,2] + d_point_cov[i,2,2]*d_T[2,2]**2

     
        d_out[i,1,0] = d_out[i,0,1] 
        d_out[i,2,1] = d_out[i,1,2] 
        d_out[i,2,0] = d_out[i,0,2] 

def get_global_cov(point_cov, T_global, T_cov):
    n=len(point_cov)
    d_point_cov = cuda.to_device(point_cov)
    d_T = cuda.to_device(T_global)
    d_T_cov = cuda.to_device(T_cov)
    d_out=cuda.device_array((n,3,3),dtype=(np.float64))
    thread=TPB
    blocks=(n+TPB-1)//TPB
    global_cov_kernel[blocks, thread](d_out, d_point_cov, d_T, d_T_cov)
    return d_out.copy_to_host()

    
TAG_SIZE=0.06 #meter
grey_color = 153
width = 640
height = 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align = rs.align(rs.stream.color)

cfg=pipeline.start(config)
cam_param=get_rs_param(cfg)
depth_sensor = cfg.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 100 #meter
clipping_distance = clipping_distance_in_meters / depth_scale

frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
color_frame = aligned_frames.get_color_frame()

depth_image = np.asanyarray(aligned_depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
# depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
# bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
# images = np.hstack((bg_removed, depth_colormap))
pipeline.stop()
#%%
at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
result=at_detector.detect(gray, estimate_tag_pose=True, tag_size=TAG_SIZE, 
				camera_params=cam_param)


frames=[]
for r in result:
    xp=r.center[0]
    yp=r.center[1] 
    tag_id=r.tag_id
    R=r.pose_R
    R[:, 2]=np.cross( R[:, 0],  R[:, 1])
    R=R@np.array([[0,1,0],
                [0,0,-1],
                [-1,0,0]])
    mu=r.pose_t
    T=np.eye(4)
    T[0:3,0:3]=R
    T[0:3, 3]=mu.flatten()
    tag_frame=o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    tag_frame.transform(T)
    frames.append(tag_frame)
#%%
T_global=np.linalg.inv(T)
K = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx = cam_param[0], fy= cam_param[1], cx=cam_param[2], cy=cam_param[3] )
rgbd =  o3d.geometry.RGBDImage.create_from_color_and_depth(color = o3d.geometry.Image(color_image),
                                                              depth = o3d.geometry.Image(depth_image), depth_scale=1/depth_scale)
pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,K )
frame=o3d.geometry.TriangleMesh.create_coordinate_frame(0.25)
pc.transform(T_global)
frame.transform(T_global)
origin=o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
o3d.visualization.draw_geometries([origin, pc,frame])

#%%

Q=np.array([[20**2,0,0],
            [0, 20**2, 0],
            [0, 0, 0.01**2]])
cov=get_cloud_covariance_par(depth_image*depth_scale, Q, inv(K.intrinsic_matrix))
#cov=np.asarray([m for m in cov if not np.linalg.det(m) == 0])
cov=cov[det(cov)!=0]

# cov_test_ser = np.array([T_global[0:3,0:3]@sigma@T_global[0:3,0:3].T for sigma in cov])
cov=get_global_cov(cov, T_global, np.zeros((3,3)))
#%%
wall = o3d.geometry.TriangleMesh.create_box()
T_plane=np.array([[0.01, 0, 0, -0.005 ],
                  [0   , 1,0,-0.5],
                  [0   , 0, 1,-0.5],
                  [0   ,0,0,1]])
wall.transform(T_plane)

# o3d.visualization.draw_geometries([pc,wall, frame])

ref = wall.sample_points_uniformly(number_of_points=20000, use_triangle_normal=True)
o3d.visualization.draw_geometries([origin, pc,ref, frame])

ref_points=np.asarray(ref.points)
ref_normal=np.asarray(ref.normals)
ref_tree=KDTree(ref_points)


#%%
points=np.asarray(pc.points)
t=time.time()
_, corr = ref_tree.query(points, 1)
print("nearest neighbor search", time.time()-t)
#%%
n = [1,0,0]
mu=[0,0,0]
points=np.asarray(pc.points)
thres=0.05
normal=n

# md0=np.zeros(len(points))
# md1=np.zeros(len(points))
# t=time.time()
# for i,point in enumerate(points):    
#     c=normal@mu
#     if normal@point<=c:
#         d0 = 0 
#     else:
#         d0 = (n@mu-n@point)**2/(n@cov[i,:,:]@n)

#     if normal@point>=c+thres:
#         d1 = 0
#     else:
#         d1 = (thres + n@mu - n@point)**2/(n@cov[i,:,:]@n)
    
#     md0[i]=d0
#     md1[i]=d1
# print("serial:" ,time.time()-t)   

#%%
t=time.time()
normals=ref_normal[corr]
mus=ref_points[corr]
mds=get_md_par(points,mus , thres , cov, normals)
print("m-dost parallel: ",time.time()-t)   
#%%
t=time.time()
n_sample = np.zeros(len(ref_points))
md_ref = np.zeros((len(ref_points),2))
for i, idx in enumerate(corr):
    n_sample[idx]+=1
    md_ref[idx,0]+=mds[i,0]
    md_ref[idx,1]+=mds[i,1]

chi_0=chi2.sf(md_ref[:,0], n_sample)
chi_1=chi2.sf(md_ref[:,1], n_sample)

chi_0[np.isnan(chi_0)] = 0.5
chi_1[np.isnan(chi_1)] = 0.5

print("chi2: ",time.time()-t)   

#%%
chi=np.array([chi_1[i]/(chi_1[i]+chi_0[i]) for i in range(len(chi_1))])
# chi=np.array([mds[i,0]/(mds[i,0]+mds[i,1]) for i in range(len(chi_1))])
# chi=np.array([1 for i in range(len(chi_1))])

color=(chi*255).astype(np.uint8)
color=cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
color = np.squeeze(color)
ref.colors = o3d.utility.Vector3dVector(color/255)
# color=0.8*np.asarray(pc.colors) + 0.2*color/255
# pc_heatmap=deepcopy(pc)
# pc_heatmap.colors=o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([ref,origin, pc,frame])
