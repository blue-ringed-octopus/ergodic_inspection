# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:30:00 2024

@author: hibad
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:02:45 2024

@author: hibad
"""
import sys
import cv2
import numpy as np
from numpy import sin, cos, arccos, trace, arctan2
from numpy.linalg import norm, inv
import pyrealsense2 as rs
from pupil_apriltags import Detector
import os
os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")
import matplotlib.pyplot as plt
import time
sys.path.append('../')
from Lie import SO3, SE3


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


def angle_wrapping(theta):
    return arctan2(sin(theta), cos(theta))

#%%
WINDOW_SCALE=1
TAG_SIZE=0.06 #meter
screenWidth=640; #pixel
screenHeight=480; #pixel
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, screenWidth, screenHeight, rs.format.bgr8, 30)

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
K_inv = inv(K)
T_c_to_r = np.array([[0,0,1,0],
                     [-1,0,0,0],
                     [0,-1,0,0],
                     [0,0,0,1]])
mu=np.zeros(3)
sigma=np.zeros((3,3))
R=np.eye(3)
R[0,0]=0.01
R[1,1]=0.01
R[2,2]=0.1

Q=np.eye(6)
Q[0,0]=1**2 # x pixel
Q[1,1]=1**2 # y pixel
Q[2,2]=1**2  # depth
Q[3:6, 3:6] *= (np.pi/2)**2 #axis angle
landmarks={}


#%%
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# # Convert images to numpy arrays
rgb = np.asanyarray(color_frame.get_data())
#Tag detection
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
result=at_detector.detect(gray, estimate_tag_pose=True, tag_size=TAG_SIZE, 
				camera_params=cam_param)


features={}
for r in result:
    xp=r.center[0]
    yp=r.center[1] 
    z=r.pose_t.flatten()[2]
    Rot=r.pose_R
    Rot[:, 2]=np.cross(Rot[:, 0], Rot[:, 1])
    t= r.pose_t
    Rot=Rot@np.array([[0,1,0],
                    [0,0,-1],
                    [-1,0,0]]) #rotate such that x-axis points outward, z-axis points upward 
    M=np.eye(4)
    M[0:3,0:3] = Rot
    M[0:3,3] = np.squeeze(t)
    features[r.tag_id]= {"xp": xp, "yp": yp, "z":z,  "M": M}
    


def initialize_new_landmarks(features, mu, sigma, landmarks):
    T=SE3.Exp([mu[0], mu[1], 0,0,0, mu[2]])@T_c_to_r     #coordinate transformation from camera coordinate to world coordinate
    for landmark_id in features:
        if not landmark_id in landmarks.keys():
            landmark=features[landmark_id]
            loc=landmark['z']*K_inv@np.array([landmark["xp"], landmark["yp"],1])  
            loc=T@np.hstack((loc,[1]))           
            loc=loc[0:3]          
            r=landmark["M"][0:3,0:3] #feature orientation in camera frame
            r=T[0:3,0:3]@r  #feature orientation in world frame
            theta=SO3.Log(r)
            theta[0:2]=[0,0] # remove rotations in x,y axis
            r=SO3.Exp(theta)
            M=np.eye(4)
            M[0:3,0:3] = r
            M[0:3,3] = loc
            tau = SE3.Log(M)
            zeta=np.hstack((tau[0:3], tau[5])) #only take the z rotation
            landmarks[landmark_id]=mu.shape[0]
            mu=np.hstack((mu.copy(),zeta))
            sigma_new=np.diag(np.ones(sigma.shape[0]+len(zeta))*99999999999)
            sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
            sigma=sigma_new
    return mu, sigma,  landmarks      
            
mu, sigma, landmarks = initialize_new_landmarks(features, mu, sigma, landmarks)

tau_r=[mu[0], mu[1], 0, 0,0, mu[2]]
fr = np.zeros((6,3))
fr[0,0]=1
fr[1,1]=1
fr[5,2]=1
ftag = np.zeros((6,4))
ftag[0,0]=1
ftag[1,1]=1
ftag[2,2] = 1
ftag[5,3] = 1

T_c_to_w=SE3.Exp(tau_r)@T_c_to_r
T_w_to_c=inv(T_c_to_w)
dmu=np.zeros(mu.shape)
for feature_id in features:    
    feature=features[feature_id]
    idx=landmarks[feature_id]
    tau_l_hat=mu[idx:idx+4].copy() #global feature location
    tau_l = np.array([tau_l_hat[0], tau_l_hat[1], tau_l_hat[2], 0, 0, tau_l_hat[3]])
    M_tag = SE3.Exp(tau_l)
    
    M_tag_c = feature["M"]
    M_tag_c_bar = T_w_to_c@M_tag

    tau_bar= SE3.Log(M_tag_c_bar)
    dtau = SE3.Log(M_tag_c) - tau_bar#measurement error 
    dtau = SE3.Log(SE3.Exp(dtau))
    
    J_cr=np.zeros((6,6))
    J_cr[0:3,0:3] = T_c_to_r[0:3,0:3].T
    J_cr[3:6,3:6] = T_c_to_r[0:3,0:3].T
    J_cr[0:3,3:6] = -T_c_to_r[0:3,0:3].T@SO3.hat(T_c_to_r[0:3,3])
    
    
    jr=-SE3.Jl_inv(tau_bar)@J_cr@SE3.Jr(tau_r)@fr #jacobian of robot pose
    jtag=SE3.Jr_inv(tau_bar)@SE3.Jr(tau_l)@ftag   #jacobian of tag pose
        
    H=np.zeros((6,7)) #number of obervation: 6, number of state:7 
    H[0:6, 0:3] = jr
    H[0:6:, 3:7] = jtag
    
    F=np.zeros((7,mu.shape[0]))
    F[0:3,0:3]=np.eye(3)
    F[3:7, idx:idx+4]=np.eye(4) 

    H=H@F
    k=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
    dmu+=k@(dtau)
    sigma=(np.eye(mu.shape[0])-k@H)@(sigma)

mu=mu+dmu
sigma[0:3, 0:3]=sigma[0:3,0:3]+R

x_r=SE3.Exp([mu[0], mu[1],0, 0,0, mu[2]])
plt.arrow(x_r[0,3], x_r[1,3], 0.1*cos(mu[2]), 0.1*sin(mu[2]))
for i in landmarks.values():
    x_tag = SE3.Exp([mu[i], mu[i+1], mu[i+2],0,0, mu[i+3]])
    plt.arrow(x_tag[0,3], x_tag[1,3], 0.1*cos(mu[i+3]), 0.1*sin(mu[i+3]))
plt.axis('scaled')
plt.xlim([0.0, 0.8])
plt.ylim([-0.5, 0.5])