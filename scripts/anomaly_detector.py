#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:26:02 2024

@author: hibad
"""

from scipy.stats import chi2, norm
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
from numpy.linalg import norm
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

        if npoint <= nmu:
            d0 = 0
        else:
            d0 = (nmu-npoint)**2/ncn

        if npoint >= nmu+d_epsilon:
            d1 = 0
        else:
            d1 = (d_epsilon + nmu-npoint)**2/ncn

        d_out[i, 0] = d0
        d_out[i, 1] = d1


def get_md_par(points, mu, epsilon, cov, normal):
    n = points.shape[0]
    d_mu = cuda.to_device(mu)
    d_cov = cuda.to_device(cov)
    d_normal = cuda.to_device(normal)
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


class Anomaly_Detector:
    def __init__(self, mesh, bounding_box, thres=1):
        self.mesh = mesh
        num_points = 100000
        self.num_points = num_points
        pc = mesh.sample_points_uniformly(
            number_of_points=num_points, use_triangle_normal=True)
        # pc = pc.crop(bounding_box)
        # pc.paint_uniform_color([0,0,0])
        self.bounding_box = bounding_box
        self.reference = pc
        self.p_anomaly = np.ones(len(pc.points))*0.5
        self.ref_normal = np.asarray(pc.normals)
        self.ref_points = np.asarray(pc.points)
        self.ref_tree = KDTree(self.ref_points)
        self.neighbor_count = 40
        _, corr = self.ref_tree.query(self.ref_points, k=self.neighbor_count)

        self.self_neighbor = corr
        self.thres = thres
        self.n_sample = np.zeros(num_points)
        self.md_ref = np.zeros((num_points, 2))
        self.chi2 = np.zeros((num_points, 2))
  #  def detect_thread(self, node):

    def paint_pc(self, pc, mds):
        c = np.array([mds[i, 0]/(mds[i, 0]+mds[i, 1])
                     for i in range(len(mds))])
        #chi=np.array([1 for i in range(len(chi_1))])

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

    def paint_ref(self, c):
        pc = deepcopy(self.reference)
        color = (c*255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = np.squeeze(color)
        pc.colors = o3d.utility.Vector3dVector(color/255)

        return pc

    def calculate_md(self, node):
        pass

    def sum_md(self, mds, corr):
        # t = time.time()
        # count = Counter(corr)        
        # n_sample = np.zeros(self.num_points)
        # n_sample[list(count.keys())] = list(count.values())
        # self.n_sample = n_sample
        # n_sample = n_sample[self.self_neighbor]
        # self.n_sample = np.sum(n_sample, 1)
         
        one = np.ones(len(mds))*3
        for i in range(self.neighbor_count):
            np.add.at(self.md_ref, self.self_neighbor[corr, :][:, i], mds)
            np.add.at(self.n_sample , self.self_neighbor[corr,:][:,i], one)
        # n_sample = np.zeros(self.num_points)
        # for i, idx in enumerate(corr):
        #     n_idx = self.self_neighbor[idx, :]
        #     n_sample[n_idx] += 1
            # n_sample[idx] += 1
            # self.md_ref[n_idx,:] += mds[i,:]
            # self.md_ref[idx,:] += mds[i,:]
        # print("sum", np.max(np.abs(self.n_sample - n_sample)))
        # print("sum md time", time.time()-t)

    def ICP(self, pc):
        result_icp = o3d.pipelines.registration.registration_icp(
            pc, self.reference , 10, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = result_icp.transformation
        
        return pc.transform(T), T
    
    def detect(self, node):
        print("estimating anomaly: node " + str(node.id))

        node_pose = node.M.copy()
        cloud = node.local_map['pc'].copy()
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(cloud["points"])
        p = p.transform(node_pose)
        p, T = self.ICP(p)
        p = p.uniform_down_sample(200)
        point_cov = node.local_map['cov'].copy()
        point_cov = point_cov[np.arange(0,len(point_cov), 200)]
        sigma_node = np.zeros((3,3))#node.cov
        points = np.asarray(p.points)

        cov=get_global_cov(point_cov, T@node_pose, sigma_node)
        # self.cov=get_global_cov(point_cov, T@node_pose, sigma_node)
        # cov=self.cov/100
        # cov = [np.eye(3)*0.001 for _ in range(len(points))]

        _, corr = self.ref_tree.query(points, k=1)

        normals = self.ref_normal[corr]
        mus = self.ref_points[corr]
        mds = get_md_par(points, mus, self.thres, cov, normals)
        p = self.paint_pc(p, mds)
        # p = self.paint_cov(p, cov)
        self.sum_md(mds, corr)
        idx = self.n_sample>0
        # z_nominal = (self.md_ref[idx, 0] - self.n_sample[idx])/np.sqrt(2*self.n_sample[idx])
        # z_anomaly = (self.md_ref[idx, 1] - self.n_sample[idx])/np.sqrt(2*self.n_sample[idx])
        chi2_nominal = np.nan_to_num(chi2.sf(self.md_ref[idx, 0], self.n_sample[idx]), nan=0.5)
        chi2_anomaly = np.nan_to_num(chi2.sf(self.md_ref[idx, 1], self.n_sample[idx]), nan=0.5)
        # chi2_nominal_test = norm.sf(z_nominal)
        # chi2_anomaly_test =norm.sf(z_anomaly)
        
        # print(np.max(np.abs(chi2_nominal - chi2_nominal_test)))
        # print(np.max(np.abs(chi2_anomaly - chi2_anomaly_test)))

        p_nominal = (chi2_nominal + 0.00000000) * (1-self.p_anomaly[idx])
        p_anomaly = (chi2_anomaly + 0.00000000) * self.p_anomaly[idx]
        p_anomaly = p_anomaly/(p_nominal + p_anomaly)
        self.p_anomaly[idx] = p_anomaly
        # self.p_anomaly[idx] = chi2_anomaly_test
        ref = self.paint_ref(self.p_anomaly)
        self.md_ref = np.zeros((self.num_points,2))
        self.n_sample = np.zeros(self.num_points)
        # self.chi2= np.zeros((self.num_points,2))
        return p.crop(self.bounding_box), ref.crop(self.bounding_box)
