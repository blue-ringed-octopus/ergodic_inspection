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

# def draw_frame(img, tag, K):
#     img=cv2.circle(img, (int(tag["xp"]), int(tag["yp"])), 5, (0, 0, 255), -1)
#     M=tag["M"].copy()
    
#     x_axis=K@M[0:3,:]@np.array([0.06,0,0,1])
#     x_axis=x_axis/(x_axis[2])
    
#     img=cv2.arrowedLine(img, (int(tag["xp"]), int(tag["yp"])), (int(x_axis[0]), int(x_axis[1])), 
#                                      (0,0,255), 5)  
#     return img


class EKF:
    def __init__(self, node_id, R, Q ,  T_c_to_r, K, odom, tag_size, tag_family = "tag36h11", fixed_landmarks=[]):
        self.fixed_landmarks = fixed_landmarks
        self.features={}
        self.tag_size = tag_size
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
        # self.R[5,5] *= (np.pi/2)**2
        self.R = R
        #observation covariance
        # self.Q=np.eye(6)
        # self.Q[0,0]=1**2 # 
        # self.Q[1,1]=1**2 # 
        # self.Q[2,2]=1**2 #
        # self.Q[3:6, 3:6] *= (np.pi/2)**2 #axis angle
        self.Q = Q
        self.Q_img = np.eye(3)
        self.Q_img[0,0]=1**2 #x-pixel
        self.Q_img[1,1]=1**2 #y-pixel
        self.Q_img[2,2]=0.001**2 #depth 
        
        self.at_detector = Detector(
                    families=tag_family,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                    )        
        self.odom_prev = odom   
        
    def reset(self, node_id, pc_info, landmarks={}, fixed_landmarks=None):
        if not fixed_landmarks == None:
            self.fixed_landmarks = fixed_landmarks
        self.id = node_id
        self.mu=[np.eye(4)]
        self.sigma=np.zeros((6,6))            
        self.features={}
        self.landmarks=landmarks
        self.cloud = {"pc": {"points": [], "colors":[] }, "cov": [], "depth": [], "rgb": [], "features": {} ,"cam_param": self.K.copy(), "cam_transform": self.T_c_to_r.copy()}
        
        if not pc_info == None:
            self._process_pointcloud(pc_info)
        
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
        result=self.at_detector.detect(gray, estimate_tag_pose=True, tag_size=self.tag_size, 
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
            if (not feature_id in self.features.keys()) and (not feature_id in self.fixed_landmarks):
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
 
        for i,feature_id in enumerate(features):  
            # udpate robot pose based on ground truth landmarks
            if feature_id in self.fixed_landmarks:
                feature=features[feature_id]
                
                #ground truth feature location
                M_tag = self.landmarks[feature_id]
                Z_bar = inv(mu[0])@M_tag  #feature location in camera frame
                z_bar = SE3.Log(Z_bar)
          
                #observed feature pose
                Z = feature["M"]
                z = SE3.Log(Z)
                
                #measurement error 
                dz[6*i:6*i+6] = SE3.Log(SE3.Exp(z - z_bar)) 
    
                Jr=-SE3.Jl_inv(z_bar) #jacobian of robot pose
                
                #number of obervation: 6, number of local state:6 
                h=np.zeros((6,6))
                h[0:6, 0:6] = Jr
                
                #number local state, number of global state 
                F=np.zeros((6,6*len(mu)))
                F[0:6,0:6]=np.eye(6)    
                
                H[6*i:6*i+6,:] += h@F
                Q[6*i:6*i+6, 6*i:6*i+6] =self.Q.copy()
                
            # udpate robot pose and features simultaneously     
            else:
                feature=features[feature_id]
                idx=self.features[feature_id]
                
                #global feature location
                M_tag_bar = mu[idx].copy() 
                Z_bar = inv(mu[0])@M_tag_bar  #feature location in camera frame
                z_bar = SE3.Log(Z_bar)
          
                #observed feature pose
                Z = feature["M"]
                z = SE3.Log(Z)
                
                #measurement error 
                dz[6*i:6*i+6] = SE3.Log(SE3.Exp(z - z_bar))
    
                Jr=-SE3.Jl_inv(z_bar) #jacobian of robot pose
                Jtag=SE3.Jr_inv(z_bar)   #jacobian of tag pose
                
                #number of obervation: 6, number of local state:12 
                h=np.zeros((6,12))
                h[0:6, 0:6] = Jr
                h[0:6:, 6:12] = Jtag
                
                #number local state, number of global state 
                F=np.zeros((12,6*len(mu)))
                F[0:6,0:6]=np.eye(6)
                F[6:12, 6*idx:6*idx+6]=np.eye(6) 
    
                
                H[6*i:6*i+6,:] += h@F
                Q[6*i:6*i+6, 6*i:6*i+6] =self.Q.copy()
            
        K=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
        sigma=(np.eye(len(mu)*6)-K@H)@(sigma)
        dmu=K@(dz)
        sigma[6:, 6:] += np.eye(len( sigma[6:, 6:]))* 0.001
        if not np.isnan(dmu).all():
            for i in range(len(mu)):
                self.mu[i]=mu[i]@SE3.Exp(dmu[6*i:6*i+6])
            self.sigma=(sigma+sigma.T)/2
        else:
            print("nan correntions")
    def camera_update(self, rgb, depth):    
        features=self._detect_apriltag(rgb, depth, 2)
        # for feature in features.values():
        #     rgb=draw_frame(rgb, feature, self.K)
        self._initialize_new_features(features)
        self._correction(features)
        return features.copy()
    
    def motion_update(self, odom, Rv):
        #get relative transformation
        U = np.linalg.inv(self.odom_prev)@odom
        u = SE3.Log(U)

        #apply transformation
        mu=self.mu.copy()
        # M_prev=mu[0]
        # M = M_prev@U
        # mu[0] = M
        mu[0] = mu[0]@U
        F=np.zeros((6,6*len(mu)))
        F[0:6,0:6]=np.eye(6)
        
        Jx= SE3.Ad(inv(U))
        
        Jx = F.T@Jx@F
        Jx[6:,6:]=np.eye(Jx[6:,6:].shape[0])
        Ju=SE3.Jr(u)
        if not np.isnan(mu).all():
            self.mu = mu
            self.sigma=(Jx)@self.sigma@(Jx.T)+F.T@(Ju)@(self.R+0.01*self.R@Rv)@(Ju.T)@F
            self.odom_prev=odom
        else:
             print("nan motion updates")
             
    def get_posterior(self):
        pos = {"mu":  self.mu.copy(), "sigma":self.sigma.copy(),  "features": self.features.copy()}
        return pos 
    
if __name__ == "__main__":
    import yaml
    # from scipy.spatial.transform import Rotation as R
    import pyrealsense2 as rs
    import os
    os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")

    with open('../param/sim/estimation_param.yaml', 'r') as file:
        params = yaml.safe_load(file)
        
    with open('../resources/sim/prior_features.yaml', 'r') as file:
        prior =  yaml.safe_load(file)
    def get_rs_param(cfg):
        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        return [intr.fx, intr.fy, intr.ppx, intr.ppy]
    
    def draw_frame(img, R,t, K):
        T=K@np.concatenate((R,t),1)
        x=T@[0,0,0,1]
        x=x/x[2]
        xp=x[0]
        yp=x[1]
        x_axis=T@np.array([0.06,0,0,1])
        x_axis=x_axis/(x_axis[2])
        y_axis=T@np.array([0,0.06,0,1])
        y_axis=y_axis/(y_axis[2])
        z_axis=T@np.array([0,0,0.06,1])
        z_axis=z_axis/(z_axis[2])
        
      
        img=cv2.arrowedLine(img, (int(xp), int(yp)), (int(y_axis[0]), int(y_axis[1])), 
                                (0,255,0), 2)  
        img=cv2.arrowedLine(img, (int(xp), int(yp)), (int(z_axis[0]), int(z_axis[1])), 
                                  (255,0,0), 2)  
        img=cv2.circle(img, (int(xp), int(yp)), 5, (0, 0, 0), -1)
        img=cv2.arrowedLine(img, (int(xp), int(yp)), (int(x_axis[0]), int(x_axis[1])), 
                                  (0,0,255), 2)   

        return img

    
    cv2.destroyAllWindows()
    WINDOW_SCALE=1
    TAG_SIZE=0.06 #meter
    screenWidth=640; #pixel
    screenHeight=480; #pixel
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, screenWidth, screenHeight, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)
    # Start streaming
    cfg=pipeline.start(config)
    cam_param=get_rs_param(cfg)
    at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
    
    
    K=np.array([[cam_param[0], 0, cam_param[2]],
                [0, cam_param[1], cam_param[3]],
                [0,0,1]])
    landmarks={0:np.eye(4)}

    ekf = EKF(node_id=0, 
              T_c_to_r = np.eye(4), 
              K = K, 
              odom = np.eye(4), 
            R = np.array(params["EKF"]["motion_noise"]),
            Q = np.array(params["EKF"]["observation_noise"]),
            tag_size = params["EKF"]["tag_size"],
            tag_family = params["EKF"]["tag_families"],
            fixed_landmarks = [0])
    ekf.reset(0, landmarks=landmarks, fixed_landmarks = [0], pc_info=None)

    n=12
    mu=np.zeros(n)

    sigma=np.eye(n)*0
    # sigma=np.zeros((n,n))
    # sigma[3:6,3:6]*=0
 
    tags={}
    init=True
    try:
        while True:
            ekf.motion_update(np.eye(4), np.zeros((6,6)))
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            depth = np.asanyarray(aligned_depth_frame.get_data())
            rgb = np.asanyarray(color_frame.get_data())

            
            rgb_raw=rgb.copy()
            rgb_kalman=rgb.copy()

            tags = ekf.camera_update(rgb, depth)
            M_camera = ekf.mu[0]
            
            tag_M = []
            for M in ekf.mu[1:]:
                tag_M.append(inv(M_camera)@M)
            rgb_raw = cv2.putText(rgb_raw, "Raw", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
            
            for M in tag_M:
                rgb_kalman= draw_frame(rgb_kalman,M[0:3,0:3],M[0:3,3].reshape(3,1), K)
                
            rgb_kalman= draw_frame(rgb_kalman,inv(M_camera)[0:3,0:3],inv(M_camera)[0:3,3].reshape(3,1), K)

            rgb_kalman = cv2.putText(rgb_kalman, "Kalman", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)

            for tag in tags.values():
                rgb_raw = draw_frame(rgb_raw,tag["M"][0:3,0:3],tag["M"][0:3,3].reshape(3,1), K)
            img=np.hstack((rgb_raw, rgb_kalman))
                # image=cv2.circle(image, (int(x[0]),int(x[1])), radius=5, color=[0,0,255], thickness=-1)
            cv2.imshow('Raw', img)

            cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        pipeline.stop()