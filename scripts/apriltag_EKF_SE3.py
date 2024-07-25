#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:40:13 2023

@author: barc
"""
import cv2
from pupil_apriltags import Detector
import numpy as np
from numpy.linalg import inv
import time
from numba import cuda
from Lie import SE3
import open3d as o3d

TPB=32
# @cuda.jit()
# def cloud_cov_kernel(d_out, d_depth, d_Q, d_K_inv):
#     i,j = cuda.grid(2)
#     nx, ny=d_depth.shape

#     if i<nx and j<ny:
#         n=int(j+i*ny)
#         d = d_depth[i,j]
#         if not d == 0:
#             J00 = d*d_K_inv[0,0]
#             J01 = d*d_K_inv[0,1]
#             J02 = d_K_inv[0,2] + i*d_K_inv[0,0] + j*d_K_inv[0,1]
#             J10 = d*d_K_inv[1,0]
#             J11 = d*d_K_inv[1,1]
#             J12 = d_K_inv[1,2] + i*d_K_inv[1,0] + j*d_K_inv[1,1]
#             J20 = d*d_K_inv[2,0]
#             J21 =  d*d_K_inv[2,1]
#             J22 = d_K_inv[2,2] + i*d_K_inv[2,0] + j*d_K_inv[2,1]
        
#             d_out[n,0,0] = d_Q[0,0]*J00**2 + d_Q[1,1]*J01**2 + d_Q[2,2]*J02**2
#             d_out[n,0,1] = d_Q[0,0]*J00*J10 + d_Q[1,1]*J01*J11 + d_Q[2,2]*J02*J12
#             d_out[n,0,2] = d_Q[0,0]*J00*J20 + d_Q[1,1]*J01*J21 + d_Q[2,2]*J02*J22
            
#             d_out[n,1,0] = d_out[n,0,1] 
#             d_out[n,1,1] = d_Q[0,0]*J10**2 + d_Q[1,1]*J11**2 + d_Q[2,2]*J12**2
#             d_out[n,1,2] = d_Q[0,0]*J10*J20 + d_Q[1,1]*J11*J21 + d_Q[2,2]*J12*J22
            
#             d_out[n,2,0] = d_out[n,0,2]
#             d_out[n,2,1] = d_out[n,1,2]
#             d_out[n,2,2] = d_Q[0,0]*J20**2 + d_Q[1,1]*J21**2 + d_Q[2,2]*J22**2
@cuda.jit()
def cloud_cov_kernel(d_out, d_depth, d_Q, d_K_inv, d_T):
    i,j = cuda.grid(2)
    nx, ny=d_depth.shape

    if i<nx and j<ny:
        n=int(j+i*ny)
        d = d_depth[i,j]
        if not d == 0:
            J00 = d*(d_K_inv[0,0]*d_T[0,0] +d_K_inv[1,0]*d_T[0,1])
            J01 = d*(d_K_inv[0,1]*d_T[0,0] +d_K_inv[1,1]*d_T[0,1])
            J02 = d_T[0,2] +d_K_inv[0,2]*d_T[0,0] + d_K_inv[1,2]*d_T[0,1] + i*(d_K_inv[0,0]*d_T[0,0] +d_K_inv[1,0]*d_T[0,1]) + j*(d_K_inv[0,1]*d_T[0,0] +d_K_inv[1,1]*d_T[0,1])
            J10 = d*(d_K_inv[0,0]*d_T[1,0] +d_K_inv[1,0]*d_T[1,1])
            J11 = d*(d_K_inv[0,1]*d_T[1,0] +d_K_inv[1,1]*d_T[1,1])
            J12 = d_T[1,2] +d_K_inv[0,2]*d_T[1,0] + d_K_inv[1,2]*d_T[1,1] + i*(d_K_inv[0,0]*d_T[1,0] +d_K_inv[1,0]*d_T[1,1]) + j*(d_K_inv[0,1]*d_T[1,0] +d_K_inv[1,1]*d_T[1,1])
            J20 = d*(d_K_inv[0,0]*d_T[2,0] +d_K_inv[1,0]*d_T[2,1])
            J21 = d*(d_K_inv[0,1]*d_T[2,0] +d_K_inv[1,1]*d_T[2,1])
            J22 = d_T[2,2] +d_K_inv[0,2]*d_T[2,0] + d_K_inv[1,2]*d_T[2,1] + i*(d_K_inv[0,0]*d_T[2,0] +d_K_inv[1,0]*d_T[2,1]) + j*(d_K_inv[0,1]*d_T[2,0] +d_K_inv[1,1]*d_T[2,1])
        
            d_out[n,0,0] = d_Q[0,0]*J00**2 + d_Q[1,1]*J01**2 + d_Q[2,2]*J02**2
            d_out[n,0,1] = d_Q[0,0]*J00*J10 + d_Q[1,1]*J01*J11 + d_Q[2,2]*J02*J12
            d_out[n,0,2] = d_Q[0,0]*J00*J20 + d_Q[1,1]*J01*J21 + d_Q[2,2]*J02*J22
            
            d_out[n,1,0] = d_out[n,0,1] 
            d_out[n,1,1] = d_Q[0,0]*J10**2 + d_Q[1,1]*J11**2 + d_Q[2,2]*J12**2
            d_out[n,1,2] = d_Q[0,0]*J10*J20 + d_Q[1,1]*J11*J21 + d_Q[2,2]*J12*J22
            
            d_out[n,2,0] = d_out[n,0,2]
            d_out[n,2,1] = d_out[n,1,2]
            d_out[n,2,2] = d_Q[0,0]*J20**2 + d_Q[1,1]*J21**2 + d_Q[2,2]*J22**2
            
def get_cloud_covariance_par(depth, Q, K_inv, T):
    nx, ny = depth.shape
    d_depth = cuda.to_device(depth)
    d_Q = cuda.to_device(Q)
    d_K_inv = cuda.to_device(K_inv)
    d_T = cuda.to_device(T)
    d_out=cuda.device_array((nx*ny, 3, 3),dtype=(np.float64))
    thread=(TPB, TPB)
    blocks=((nx+TPB-1)//TPB,(ny+TPB-1)//TPB)
    cloud_cov_kernel[blocks, thread](d_out, d_depth,d_Q, d_K_inv, d_T)
    cov=d_out.copy_to_host()
    return cov


np.set_printoptions(precision=2)

def draw_frame(img, tag, K):
    img=cv2.circle(img, (int(tag["xp"]), int(tag["yp"])), 5, (0, 0, 255), -1)
    M=tag["M"].copy()
    
    x_axis=K@M[0:3,:]@np.array([0.06,0,0,1])
    x_axis=x_axis/(x_axis[2])
    
    img=cv2.arrowedLine(img, (int(tag["xp"]), int(tag["yp"])), (int(x_axis[0]), int(x_axis[1])), 
                                     (0,0,255), 5)  
    return img


class EKF:
    def __init__(self, node_id, T_c_to_r, K, odom):
        self.features={}

        self.T_c_to_r=T_c_to_r
        self.T_r_to_c=inv(T_c_to_r)
        self.K = K
        self.K_inv=np.linalg.inv(self.K)
        self.t=time.time()

        #motion covariance
        # self.R=np.eye(6)
        # self.R[0,0]=0.01 #x
        # self.R[1,1]=0.01 #y
        # self.R[2,2]=0.0001 #z
        # self.R[3:5, 3:5] *= 0.0001
        # self.R[5,5] *= (np.pi/4)**2
        self.R=np.eye(6)
        self.R[0,0]=999 #x
        self.R[1,1]=999 #y
        self.R[2,2]=999 #z
        self.R[3:5, 3:5] *= 999
        self.R[5,5] *= 999

        #observation covariance
        self.Q=np.eye(6)
        self.Q[0,0]=1**2 # 
        self.Q[1,1]=1**2 # 
        self.Q[2,2]=1**2 #
        self.Q[3:6, 3:6] *= (np.pi/2)**2 #axis angle
        
        self.Q_img = np.eye(3)
        self.Q_img[0,0]=1**2 #x-pixel
        self.Q_img[1,1]=1**2 #y-pixel
        self.Q_img[2,2]=0.001**2 #depth 
        
        self.at_detector = Detector(
                    families="tag36h11",
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                    )        
        self.odom_prev = odom   
        
    def reset(self, node_id, pc_info, landmarks={}):
        print("reseting EKF")
        self.id = node_id
        self.mu=[np.eye(4)]
        self.sigma=np.zeros((6,6))
        self.features={}
        self.landmarks=landmarks
        self._process_pointcloud(pc_info)
        # self._initialize_landmarks(landmark)
        print("EKF initialized")
    
    # def _initialize_landmarks(self, landmarks):
    #    mu=self.mu.copy()       #current point estimates 
    #    sigma=self.sigma.copy() #current covariance
    #    feature_map = self.features.copy()
    #    for landmarks_id, M in landmarks.items():         
    #         feature_map[landmarks_id]=len(mu)
    #         mu.append(M)
    #         sigma_new=np.diag(np.ones(sigma.shape[0]+6)*99999999)
    #         sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
    #         sigma=sigma_new
               
    #    self.sigma=sigma
    #    self.mu=mu
    #    self.features = feature_map 
        
    def _process_pointcloud(self, pc_info):
        cloud, depth, pc_img = pc_info
        K_inv = np.ascontiguousarray(self.K_inv.copy())
        T = np.ascontiguousarray(self.T_c_to_r[0:3,0:3].copy()) #world coordinate
        cloud_cov = get_cloud_covariance_par(np.ascontiguousarray(depth),  np.ascontiguousarray(self.Q_img), K_inv, T)
        indx=~np.isnan(depth.reshape(-1))
        
        cloud["points"]=cloud["points"][indx]
        cloud["colors"]=cloud["colors"][indx]

        p=o3d.geometry.PointCloud()
        p.points=o3d.utility.Vector3dVector(cloud["points"])
        p.transform(self.T_c_to_r)
        cloud["points"]=np.asarray(p.points)
        
        cloud_cov = cloud_cov[indx]
        features = self._detect_apriltag(pc_img, depth, np.inf)
        self.cloud = {"pc": cloud,"cov": cloud_cov, "depth": depth, "rgb": pc_img, "features": features ,"cam_param": self.K.copy(), "cam_transform": self.T_c_to_r.copy()}
  
        
    def _detect_apriltag(self,rgb, depth, max_depth = 1):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        result=self.at_detector.detect(gray, estimate_tag_pose=True, tag_size=0.16, 
        				camera_params=[self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        features={}
        for r in result:
            xp=r.center[0]
            yp=r.center[1] 
            # z=depth[int(yp), int(xp)]
            z=r.pose_t.flatten()[2]
            R=r.pose_R
            R[:, 2]=np.cross(R[:, 0], R[:, 1])
            
            R=R@np.array([[0,1,0],
                            [0,0,-1],
                            [-1,0,0]]) #rotate such that x-axis points outward, z-axis points upward 
            M = np.eye(4)
            M[0:3,0:3] = R
            M[0:3, 3] = np.squeeze(r.pose_t)
            if z<max_depth:
                features[r.tag_id]= {"xp": xp, "yp": yp, "z":z, "M":self.T_c_to_r@M }
        return features
    
    def _initialize_new_features(self, features):
        mu=self.mu.copy()       #current point estimates 
        sigma=self.sigma.copy() #current covariance
        feature_map = self.features.copy()
        for feature_id in features:
            if not feature_id in self.features.keys():
                if feature_id in self.landmarks.keys():
                    M = self.landmarks[feature_id]
                else:
                    feature=features[feature_id]
                    M = mu[0]@feature["M"].copy() #feature orientation in world frame 
                
                feature_map[feature_id]=len(mu)
                mu.append(M)
                sigma_new=np.diag(np.ones(sigma.shape[0]+6)*99999999999)
                sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
                sigma=sigma_new
                
        self.sigma=sigma
        self.mu=mu
        self.features = feature_map

    def _correction(self,features):
        mu=self.mu.copy()
        sigma=self.sigma.copy()
        
        n = len(features)
        H=np.zeros((6*n,6*len(mu)))
        Q=np.zeros((6*n,6*n))
        dz = np.zeros(6*n)
        dmu = np.zeros(6)
        for i,feature_id in enumerate(features):    
            feature=features[feature_id]
            idx=self.features[feature_id]
            
            #global feature location
            M_tag_bar = mu[idx].copy() 
            Z_bar = inv(mu[0])@M_tag_bar  #feature location in camera frame
      
            Z = feature["M"]
            dmu += SE3.Log(M_tag_bar@inv(Z))
        self.mu[0] = SE3.Exp(dmu/n)
        # for i,feature_id in enumerate(features):    
        #     feature=features[feature_id]
        #     idx=self.features[feature_id]
            
        #     #global feature location
        #     M_tag_bar = mu[idx].copy() 
        #     Z_bar = inv(mu[0])@M_tag_bar  #feature location in camera frame
        #     z_bar = SE3.Log(Z_bar)
      
        #     Z = feature["M"]
        #     z = SE3.Log(Z)
            
        #     dz[6*i:6*i+6] = SE3.Log(SE3.Exp(z - z_bar)) #measurement error 

        #     Jr=-SE3.Jl_inv(z_bar) #jacobian of robot pose
        #     Jtag=SE3.Jr_inv(z_bar)   #jacobian of tag pose
            
        #     #number of obervation: 6, number of local state:12 
        #     h=np.zeros((6,12))
        #     h[0:6, 0:6] = Jr
        #     h[0:6:, 6:12] = Jtag
            
        #     #number local state, number of global state 
        #     F=np.zeros((12,6*len(mu)))
        #     F[0:6,0:6]=np.eye(6)
        #     F[6:12, 6*idx:6*idx+6]=np.eye(6) 

            
        #     H[6*i:6*i+6,:] += h@F
        #     Q[6*i:6*i+6, 6*i:6*i+6] =self.Q.copy()
            
        # K=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
        # sigma=(np.eye(len(mu)*6)-K@H)@(sigma)
        # dmu=K@(dz)
        # for i in range(len(mu)):
        #     self.mu[i]=mu[i]@SE3.Exp(dmu[6*i:6*i+6])
        
        # self.sigma=(sigma+sigma.T)/2
    
    def camera_update(self, rgb, depth):    
        features=self._detect_apriltag(rgb, depth, 2)
        for feature in features.values():
            rgb=draw_frame(rgb, feature, self.K)
        self._initialize_new_features(features)
        self._correction(features)
            
    def motion_update(self, odom, Rv):
        return 
        #get relative transformation
        U = np.linalg.inv(self.odom_prev)@odom
        u = SE3.Log(U)
        
        #apply transformation
        mu=self.mu.copy()
        M_prev=mu[0]
        M = M_prev@U
        mu[0] = M
        
        F=np.zeros((6,6*len(mu)))
        F[0:6,0:6]=np.eye(6)
        
        Jx= SE3.Ad(inv(U))
        
        Jx = F.T@Jx@F
        Jx[6:,6:]=np.eye(Jx[6:,6:].shape[0])
        Ju=SE3.Jr(u)
        self.mu = mu
        self.sigma=(Jx)@self.sigma@(Jx.T)+F.T@(Ju)@(self.R+self.R@Rv)@(Ju.T)@F
        self.odom_prev=odom
        
    def get_posterior(self):
        pos = {"mu":  self.mu.copy(), "sigma":self.sigma.copy(),  "features": self.features.copy()}
        return pos 
    
# if __name__ == "__main__":

