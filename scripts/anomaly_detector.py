#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:26:02 2024

@author: hibad
"""

import numpy as np
np.float = np.float64 
np.set_printoptions(precision=2)
import rospy
import open3d as o3d
from copy import deepcopy
import rospkg
from scipy.linalg import sqrtm
from visualization_msgs.msg import Marker, MarkerArray
import ros_numpy
from sensor_msgs.msg import PointCloud2
from numpy.linalg import inv
from scipy.spatial import KDTree
import time 
from numpy import sin, cos
from hierarchical_SLAM import *
import tf
import apriltag_EKF
from numba import cuda
from scipy.stats import chi2 
import pickle
import threading
import cv2

TPB=32

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
            d1 = (d_epsilon + nmu-npoint)**2/ncn

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
rospack=rospkg.RosPack()


class Anomaly_Detector:
    def __init__(self, mesh):
        self.mesh = mesh
        number_of_points=20000
        pc = mesh.sample_points_uniformly(number_of_points=number_of_points, use_triangle_normal=True)
        self.reference = pc 
        self.p_anomaly = np.ones(len(pc.points))*0.5
        self.ref_normal = np.asarray(pc.normals)
        self.ref_points = np.asarray(pc.points)
        self.ref_tree=KDTree(self.ref_points)
        self.thres = 0.05
        self.n_sample = np.zeros(number_of_points)
        self.md_ref = np.zeros((number_of_points,2))
        self.chi2= np.zeros((number_of_points,2))
  #  def detect_thread(self, node):
        
        
    def detect(self, node):
        print("estimating anomaly")
        t=time.time()

        node_pose=node.T.copy()
        cloud=deepcopy(node.local_map).transform(node_pose)
        point_cov=node.cloud_cov
        sigma_node=node.Cov
        points=np.asarray(cloud.points)
        
        with open('oc.pickle', 'wb') as handle:
            pickle.dump(points, handle)

        cov=get_global_cov(point_cov, node_pose, sigma_node)
        _, corr = self.ref_tree.query(points, k=1)
        normals=self.ref_normal[corr]
        mus=self.ref_points[corr]
        mds=get_md_par(points,mus , self.thres , cov, normals)
        for i, idx in enumerate(corr):
            self.n_sample[idx]+=1
            self.md_ref[idx,0]+=mds[i,0]
            self.md_ref[idx,1]+=mds[i,1]
            
        
        self.chi2[:,0] = chi2.sf(self.md_ref[:,0], self.n_sample)
        self.chi2[:,1] = chi2.sf(self.md_ref[:,1], self.n_sample)    
        print(time.time()-t)

            
def get_mesh_marker(mesh_resource):
    marker=Marker()
    marker.id = 0
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.mesh_resource = mesh_resource
    marker.type = 10
    marker.pose.orientation.x=0
    marker.pose.orientation.y=0
    marker.pose.orientation.z=0
    marker.pose.orientation.w=1
    marker.color.r = 0.2
    marker.color.g = 0.2
    marker.color.b = 0.2
    marker.color.a = 0.5
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    return marker

def pc_to_msg(pc):
    points=np.asarray(pc.points)
    colors=np.asarray(pc.colors)
    pc_array = np.zeros(len(points), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('r', np.uint32),
    ('g', np.uint32),
    ('b', np.uint32),
    ])

    pc_array['x'] = points[:,0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
    pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
    pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
    pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
    
    return pc_msg

def get_ref_pc(points, chi2):
    chi=chi2[:,1]/(chi2[:,1]+chi2[:,2])
    colors = (chi*255).astype(np.uint8)
    colors = cv2.applyColorMap(colors, cv2.COLORMAP_TURBO)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    colors = np.squeeze(colors)
    
    pc_array = np.zeros(len(points), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('r', np.uint32),
    ('g', np.uint32),
    ('b', np.uint32),
    ])
    pc_array['x'] = points[:,0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
    pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
    pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
    pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
    
    return pc_msg

    
if __name__ == "__main__":
    path = rospack.get_path("ergodic_inspection")
    mesh_resource = "file:///" + path + "/resources/ballast.STL"
    
    mesh = o3d.io.read_triangle_mesh(path+"/resources/ballast.STL")


    
    br = tf.TransformBroadcaster()
    rospy.init_node('estimator',anonymous=False)
    
    ekf=apriltag_EKF.EKF(0)
    graph_slam=Graph_SLAM(np.zeros(3), ekf)
    detector=Anomaly_Detector(mesh)
    marker = get_mesh_marker(mesh_resource)

    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
    cad_pub = rospy.Publisher("/ref", Marker, queue_size = 2)

    rate = rospy.Rate(30) 
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        cad_pub.publish(marker)

        optimized=graph_slam.update()
        
     
        plot_graph(graph_slam.front_end, factor_graph_marker_pub)
        
        mu=graph_slam.mu.copy()        
        br.sendTransform([mu[0], mu[1], 0],
                        tf.transformations.quaternion_from_euler(0, 0, mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")

        if optimized:
            detector.detect(graph_slam.front_end.pose_nodes[0])
            pc_msg=pc_to_msg(graph_slam.global_map)
            pc_pub.publish(pc_msg)
            
            ref_pc = get_ref_pc(detector.ref_points, detector.p_anomaly)
        rate.sleep()

