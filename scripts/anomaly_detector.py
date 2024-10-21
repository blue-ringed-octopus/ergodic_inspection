#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:26:02 2024

@author: hibad
"""

from scipy.stats import chi2, norm
from scipy.special import erf, erfc
import time
from collections import Counter
from Lie import SE3, SO3
import cv2
import threading
import pickle
from numba import cuda
from scipy.spatial import KDTree
from copy import deepcopy
import open3d as o3d
import numpy as np
from numpy.linalg import  inv
from  math import sqrt
from copy import deepcopy
from scipy.stats import bernoulli 

np.float = np.float64
np.set_printoptions(precision=2)
TPB = 32

@cuda.jit()
def md_kernel(d_out, d_epsilon, d_cov, d_normal, d_p, d_mu):
    i = cuda.grid(1)
    n = d_out.shape[0]
    if i < n:
        nmu = d_mu[i, 0]*d_normal[i, 0]+d_mu[i, 1] * \
            d_normal[i, 1]+d_mu[i, 2]*d_normal[i, 2]
        npoint = d_p[i, 0]*d_normal[i, 0]+d_p[i, 1] * \
            d_normal[i, 1]+d_p[i, 2]*d_normal[i, 2]
        ncn = d_cov[i, 0, 0]*d_normal[i, 0]**2 + 2*d_cov[i, 0, 1]*d_normal[i, 0]*d_normal[i, 1] + 2*d_cov[i, 0, 2]*d_normal[i, 0] * \
            d_normal[i, 2] + d_cov[i, 1, 1]*d_normal[i, 1]**2 + 2*d_cov[i, 1, 2] * \
            d_normal[i, 1]*d_normal[i, 2] + d_cov[i, 2, 2]*d_normal[i, 2]**2
        
        d0 = -(nmu-npoint)/sqrt(ncn)
        d1 = (d_epsilon + nmu - npoint)/sqrt(ncn)
        # if npoint <= nmu:
        #     d0 = 0
        # else:
        #     # d0 = (nmu-npoint)**2/ncn
        #     d0 = (nmu-npoint)/sqrt(ncn)


        # if npoint >= nmu+d_epsilon:
        #     d1 = 0
        # else:
        #     # d1 = (d_epsilon + nmu-npoint)**2/ncn
        #     d1 = (d_epsilon + nmu-npoint)/sqrt(ncn)

        d_out[i, 0] = d0
        d_out[i, 1] = d1


def get_md_par(points, mu, epsilon, cov, normal):
    n = points.shape[0]
    mu = np.ascontiguousarray(mu)
    d_mu = cuda.to_device(mu)
    d_cov = cuda.to_device(cov)
    d_normal = cuda.to_device( np.ascontiguousarray(normal))
    d_p = cuda.to_device(points)
    thread = TPB
    d_out = cuda.device_array((n, 2), dtype=(np.float64))
    blocks = (n+TPB-1)//TPB
    md_kernel[blocks, thread](d_out, epsilon, d_cov, d_normal, d_p, d_mu)
    return d_out.copy_to_host()


@cuda.jit()
def global_cov_kernel(d_out, d_point_cov, d_T, d_T_cov):
    i = cuda.grid(1)
    n = d_out.shape[0]
    if i < n:
        d_out[i, 0, 0] = d_point_cov[i, 0, 0]*d_T[0, 0]**2 + 2*d_point_cov[i, 0, 1]*d_T[0, 0]*d_T[0, 1] + 2*d_point_cov[i, 0, 2]*d_T[0, 0] * \
            d_T[0, 2] + d_point_cov[i, 1, 1]*d_T[0, 1]**2 + 2*d_point_cov[i,
                                                                          1, 2]*d_T[0, 1]*d_T[0, 2] + d_point_cov[i, 2, 2]*d_T[0, 2]**2
        d_out[i, 0, 1] = d_T[1, 0]*(d_point_cov[i, 0, 0]*d_T[0, 0] + d_point_cov[i, 0, 1]*d_T[0, 1] + d_point_cov[i, 0, 2]*d_T[0, 2]) + d_T[1, 1]*(d_point_cov[i, 0, 1]*d_T[0, 0] +
                                                                                                                                                   d_point_cov[i, 1, 1]*d_T[0, 1] + d_point_cov[i, 1, 2]*d_T[0, 2]) + d_T[1, 2]*(d_point_cov[i, 0, 2]*d_T[0, 0] + d_point_cov[i, 1, 2]*d_T[0, 1] + d_point_cov[i, 2, 2]*d_T[0, 2])
        d_out[i, 0, 2] = d_T[2, 0]*(d_point_cov[i, 0, 0]*d_T[0, 0] + d_point_cov[i, 0, 1]*d_T[0, 1] + d_point_cov[i, 0, 2]*d_T[0, 2]) + d_T[2, 1]*(d_point_cov[i, 0, 1]*d_T[0, 0] +
                                                                                                                                                   d_point_cov[i, 1, 1]*d_T[0, 1] + d_point_cov[i, 1, 2]*d_T[0, 2]) + d_T[2, 2]*(d_point_cov[i, 0, 2]*d_T[0, 0] + d_point_cov[i, 1, 2]*d_T[0, 1] + d_point_cov[i, 2, 2]*d_T[0, 2])

        d_out[i, 1, 1] = d_point_cov[i, 0, 0]*d_T[1, 0]**2 + 2*d_point_cov[i, 0, 1]*d_T[1, 0]*d_T[1, 1] + 2*d_point_cov[i, 0, 2]*d_T[1, 0] * \
            d_T[1, 2] + d_point_cov[i, 1, 1]*d_T[1, 1]**2 + 2*d_point_cov[i,
                                                                          1, 2]*d_T[1, 1]*d_T[1, 2] + d_point_cov[i, 2, 2]*d_T[1, 2]**2
        d_out[i, 1, 2] = d_T[2, 0]*(d_point_cov[i, 0, 0]*d_T[1, 0] + d_point_cov[i, 0, 1]*d_T[1, 1] + d_point_cov[i, 0, 2]*d_T[1, 2]) + d_T[2, 1]*(d_point_cov[i, 0, 1]*d_T[1, 0] +
                                                                                                                                                   d_point_cov[i, 1, 1]*d_T[1, 1] + d_point_cov[i, 1, 2]*d_T[1, 2]) + d_T[2, 2]*(d_point_cov[i, 0, 2]*d_T[1, 0] + d_point_cov[i, 1, 2]*d_T[1, 1] + d_point_cov[i, 2, 2]*d_T[1, 2])

        d_out[i, 2, 2] = d_point_cov[i, 0, 0]*d_T[2, 0]**2 + 2*d_point_cov[i, 0, 1]*d_T[2, 0]*d_T[2, 1] + 2*d_point_cov[i, 0, 2]*d_T[2, 0] * \
            d_T[2, 2] + d_point_cov[i, 1, 1]*d_T[2, 1]**2 + 2*d_point_cov[i,
                                                                          1, 2]*d_T[2, 1]*d_T[2, 2] + d_point_cov[i, 2, 2]*d_T[2, 2]**2

        d_out[i, 1, 0] = d_out[i, 0, 1]
        d_out[i, 2, 1] = d_out[i, 1, 2]
        d_out[i, 2, 0] = d_out[i, 0, 2]


def get_global_cov(point_cov, T_global, T_cov):
    n = len(point_cov)
    d_point_cov = cuda.to_device(point_cov)
    d_T = cuda.to_device(T_global)
    d_T_cov = cuda.to_device(T_cov)
    d_out = cuda.device_array((n, 3, 3), dtype=(np.float64))
    thread = TPB
    blocks = (n+TPB-1)//TPB
    global_cov_kernel[blocks, thread](d_out, d_point_cov, d_T, d_T_cov)
    return d_out.copy_to_host()

def get_global_cov_SE3(points, T_global, point_cov, T_cov):
    R = T_global[0:3, 0:3]
    n = len(points)
    cov = np.zeros((n,3,3))
    for i, point in enumerate(points):
        
        J = np.zeros((3,6))
        J[0:3,0:3] =  R
        J[0:3,3:6] =  -R@SO3.hat(point)
        cov[i,:,:] = R@point_cov[i, :, :]@R.T + J@T_cov@J.T
    return cov

def random_down_sample(point_cloud, covs):       
    n = len(point_cloud.points)
    n_ds = 5000
    if n<n_ds:
        return point_cloud, covs
    
    idx = np.random.choice(range(n), n_ds, replace = False)
    point_cloud = point_cloud.select_by_index(idx)
    covs = covs[idx]
    
    return point_cloud, covs
 
class Anomaly_Detector:
    def __init__(self, reference_cloud, region_idx, thres = 1):
        self.reference = reference_cloud
        self.anomaly_thres = thres
        self.region_idx = region_idx
        self.partition(self.region_idx)
        # box = reference_cloud.get_axis_aligned_bounding_box()
        # bound = [box.max_bound[0],box.max_bound[1], 0.5 ]
        # box.max_bound = bound
        
    def partition(self, region_idx):
        detectors = {}
        for id_, idx in region_idx.items():
            idx = np.array(idx)
            region_cloud = self.reference.select_by_index(idx)
            detectors[id_] = Local_Detector(region_cloud, self.anomaly_thres)
            
        self.detectors = detectors    
        
    def paint_ref(self, c):
        pc = deepcopy(self.reference)
        color = (c*255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.squeeze(color)
        pc.colors = o3d.utility.Vector3dVector(color/255)
        return pc

    def ICP(self, pc):
        result_icp = o3d.pipelines.registration.registration_icp(
            pc, self.reference , 0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = result_icp.transformation
        
        return pc.transform(T), T
    
    def _register_to_ref(self, node, features):
        # M = node.M.copy()
        if( len(node.local_map["features"])) == 0:
           M = node.M.copy() 
           print(node.id)
        else:
            # dm = SE3.Log(node.M)
            dm = np.zeros(6)
            for feature_id, feature in node.local_map["features"].items():
                dm  += SE3.Log(features[feature_id].M.copy()@inv(feature['M']))
            dm /= len(node.local_map["features"])
            M = SE3.Exp(dm)
            
        cloud = node.local_map['pc'].copy()
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(cloud["points"])
        
        p, cov = random_down_sample(p, node.local_map["cov"])

        p = p.transform(M)
        p, T = self.ICP(p)
        
        return p, T@M, cov
    
    def detect(self, node, features, region):
        print("estimating anomaly: node " + str(node.id))
        
        p, T, cov = self._register_to_ref(node, features)
        
        # T = np.eye(4)
        # point_cov = node.local_map['cov'].copy()
        point_cov = cov
        # p, point_cov = self.random_down_sample(p, point_cov)
        sigma_node = node.cov 
        print(sigma_node)
        points = np.asarray(p.points)
        # sigma_node = np.eye(6) 
        # sigma_node[0:3,0:3] *= 0.05
        # sigma_node[3:6,3:6] *= 0.001
        # cov=get_global_cov(point_cov, T@node_pose, sigma_node)
        cov = get_global_cov_SE3(points, T, point_cov, sigma_node)
        p_anomaly = self.detectors[region].detect(p, cov)
        if len(p_anomaly) == 0:
            return [], []
        
        return p_anomaly, self.region_idx[region]
        # with open('ref_cloud_detected.pickle', 'wb') as handle:
        #     pickle.dump({"ref_pc": self.ref_points, "p_anomaly": self.p_anomaly}, handle)
            
class Local_Detector:
    def __init__(self, pc, thres=1):
        self.bounding_box = pc.get_axis_aligned_bounding_box()
        self.get_ref_pc(pc)
        n = len(self.ref_points)
        self.ref_tree = KDTree(self.ref_points)
        
        # crop_pc = self.reference.crop(self.bounding_box)

        # _, self.crop_index = self.ref_tree.query(np.asarray(crop_pc.points),1)
        self.neighbor_count = 20

        self._calculate_self_neighbor()
        
        
        self.p_anomaly = np.ones(len(self.reference.points))*0.5
        self.thres = thres
        self.n_sample = np.zeros(n)
        self.md_ref = np.zeros((n, 2))
        self.chi2 = np.zeros((n, 2))

    def _calculate_self_neighbor(self):
        _, corr = self.ref_tree.query(self.ref_points, k=self.neighbor_count)
        self.self_neighbor = corr.astype(np.uint32)
                    
    def get_ref_pc(self, pc):
        self.reference = deepcopy(pc)
        self.ref_normal = np.asarray(pc.normals)
        self.ref_points = np.asarray(pc.points)
        self.num_points = len(self.ref_points)
        
    def _paint_pc(self, pc, mds):
        c = np.array([0.5 if mds[i, 0]+mds[i, 1]==0 else mds[i, 0]/(mds[i, 0]+mds[i, 1]  )
                     for i in range(len(mds))])

        color = (c*255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.squeeze(color)
        pc.colors = o3d.utility.Vector3dVector(color/255)
        return pc
    
    def paint_cov(self, pc, cov):
        c = np.array([norm(cov[i],"fro") for i in range(len(cov))])
        c = c/max(c)
        color = (c*255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.squeeze(color)
        pc.colors = o3d.utility.Vector3dVector(color/255)
        return pc
    
    def _sum_md(self, mds, corr):      
        one = np.ones(len(mds))
        for i in range(self.neighbor_count):
            np.add.at(self.md_ref, self.self_neighbor[corr, :][:, i], mds)
            np.add.at(self.n_sample , self.self_neighbor[corr,:][:,i], one)

    def _est_correspondence(self, points, cov):
        n = len(points)
        k = 1
        _, corr = self.ref_tree.query(points, k=k)
        normals = self.ref_normal[corr]
        mus = self.ref_points[corr]
        mds = np.zeros((k,n,2))
        if k > 1:
            for i in range(k):
                mds[i,:,:] = get_md_par(points, mus[:,i,:], self.thres, cov, normals[:,i,:])
            self.mds = mds
            idx = np.argmin(mds[:,:,0],0) #minimize null hypothesis 
            mds = mds[idx,range(len(idx)), :]
            corr = corr[range(len(idx)), idx]
        else:
            mds = get_md_par(points, mus, self.thres, cov, normals)
            
        return mds, corr
        
    def detect(self, p, cov):           
        # T = np.eye(4)        
        p = p.crop(self.bounding_box)
        # p, cov = self.random_down_sample(p, cov)

        points = np.array(p.points)
        print("num points: ", len(points))
        if len(points) == 0:
            return []

        mds, corr = self._est_correspondence(points, cov)
       
        # p = self._paint_pc(p, mds)
        # p = self.paint_cov(p, cov)
        self._sum_md(mds, corr)
        idx = self.n_sample>0
        z_nominal = norm.sf(self.md_ref[idx, 0]/np.sqrt(self.n_sample[idx]))
        z_anomaly  = norm.sf(self.md_ref[idx, 1]/np.sqrt(self.n_sample[idx]))

        p_nominal = (z_nominal + 0.05) * (1-self.p_anomaly[idx])
        p_anomaly = (z_anomaly + 0.05) * self.p_anomaly[idx]
        p_anomaly = p_anomaly/(p_nominal + p_anomaly)
        self.p_anomaly[idx] = p_anomaly
        # ref = self.paint_ref(self.p_anomaly)
        self.md_ref = np.zeros((self.num_points,2))
        self.n_sample = np.zeros(self.num_points)

        return self.p_anomaly.copy()

if __name__ == "__main__":
    from map_manager import Map_Manager
   
    manager = Map_Manager("../resources/sim/")
    detector = Anomaly_Detector(manager.reference, manager.region_idx, 0.04)
    with open('tests/graph.pickle', 'rb') as f:
        graph = pickle.load(f)
    graph.prior_factor.omega = np.eye(6)   

    for node in graph.pose_nodes.values():
        if not node.local_map == None:
    # node = graph.pose_nodes[0]
            coord= [node.M[0,3], node.M[1,3]]
            region = manager.coord_to_region(coord, 1)
            p, idx = detector.detect(node, graph.feature_nodes ,region)
            manager.set_entropy(p, idx)
        
    cloud = manager.visualize_entropy()
    o3d.visualization.draw_geometries([cloud])
