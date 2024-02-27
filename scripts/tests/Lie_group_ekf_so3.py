# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:02:45 2024

@author: hibad
"""
import cv2
import numpy as np
from numpy import sin, cos, arccos, trace, arctan2
from numpy.linalg import norm, inv
import pyrealsense2 as rs
from pupil_apriltags import Detector
import os
os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")
import matplotlib.pyplot as plt

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
def vee(W):
    return np.array([W[2,1], W[0,2], W[1,0]])

def hat(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def Log(R):
    theta=arccos((trace(R)-1)/2)
    if theta == 0:
        return np.zeros(3)
    u=theta*vee((R-R.T))/(2*sin(theta))
    return u

def Exp(u):
    theta=np.linalg.norm(u)
    if theta==0: 
        return np.eye(3)
    u=u/theta
    R=np.eye(3)+sin(theta)*hat(u)+(1-cos(theta))*np.linalg.matrix_power(hat(u),2)
    return R

def Jl_inv(w):
    t=norm(w)
    J=np.eye(3)-1/2*hat(w)+(1/t**2-(1+cos(t))/(2*t*sin(t)))*hat(w)@hat(w)
    return J

def Jr_inv(w):
    t=norm(w)
    J=np.eye(3) + 1/2*hat(w) + (1/t**2-(1+cos(t))/(2*t*sin(t))) * (hat(w)@hat(w))
    return J

def Jr(w):
    t=norm(w)  
    if t==0:
        return np.eye(3)
    J=np.eye(3)-((1-cos(t))/t**2) * hat(w) + ((t-sin(t))/t**3) * (hat(w)@hat(w))
    return J

def angle_wrapping(theta):
    return arctan2(sin(theta), cos(theta))


def v2t(x):
    return np.array([[cos(x[3]), -sin(x[3]),0, x[0]],
                     [sin(x[3]), cos(x[3]), 0, x[1]],
                     [0, 0, 1, x[2]],
                      [0, 0, 0,1]])
def t2v(T):
    return np.array([T[0,3], T[1,3], T[2,3], arctan2(T[1,0], T[0,0])])

def get_pixel_jacobian(mu, xl):
    c=cos(mu[2])
    s=sin(mu[2])
    
    jw=np.array([[-c, -s, c*(xl[1]-mu[1]) + s*(mu[0]-xl[0]) , c  , s , 0],
                 [s , -c, c*(mu[0]-xl[0]) + s*(mu[1]-xl[1]) , -s , c , 0],
                 [0 , 0 , 0                                 , 0  , 0 , 1]
                 ])
    
    jr=T_c_to_r[0:3,0:3].T

    
    return jr@jw   

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
    # z=depth[int(yp), int(xp)]
    z=r.pose_t.flatten()[2]
    Rot=r.pose_R
    Rot[:, 2]=np.cross(Rot[:, 0], Rot[:, 1])
    t= r.pose_t
    Rot=Rot@np.array([[0,1,0],
                    [0,0,-1],
                    [-1,0,0]]) #rotate such that x-axis points outward, z-axis points upward 
    features[r.tag_id]= {"xp": xp, "yp": yp, "z":z, "R": Rot, "t": np.squeeze(t)}
    
# draw_frame(rgb, R, t, K)    

# cv2.imshow('Raw', rgb)

# cv2.waitKey()
# cv2.destroyAllWindows()

def initialize_new_landmarks(features, mu, sigma, landmarks):
    T=v2t([mu[0], mu[1], 0, mu[2]])@T_c_to_r     #coordinate transformation from camera coordinate to world coordinate
    for landmark_id in features:
        if not landmark_id in landmarks.keys():
            landmark=features[landmark_id]
            loc=landmark['z']*K_inv@np.array([landmark["xp"], landmark["yp"],1])  
            loc=T@np.hstack((loc,[1]))           
            loc=loc[0:3]          
            R=landmark["R"] #feature orientation in camera frame
            R=T[0:3,0:3]@R  #feature orientation in world frame          
            tau=Log(R) #axis-angle representation        
            zeta=np.hstack((loc, tau[2])) #only take the z rotation
            landmarks[landmark_id]=mu.shape[0]
            mu=np.hstack((mu.copy(),zeta))
            sigma_new=np.diag(np.ones(sigma.shape[0]+len(zeta))*99999999999)
            sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
            sigma=sigma_new
    return mu, sigma,  landmarks      
            
mu, sigma, landmarks = initialize_new_landmarks(features, mu, sigma, landmarks)


T_c_to_w=v2t([mu[0], mu[1], 0, mu[2]])@T_c_to_r
T_w_to_c=inv(T_c_to_w)
dmu=np.zeros(mu.shape)
for feature_id in features:    
    feature=features[feature_id]
    idx=landmarks[feature_id]
    
    xl=mu[idx:idx+3].copy() #global feature location
    
    z_bar=T_w_to_c@np.concatenate((xl, [1])) #feature location in camera frame
    z_bar=z_bar[0:3]
                
    theta=angle_wrapping(mu[idx+3].copy()) #estimated planar orientation of the tag
    
    R_bar=Exp([0,0,theta])        #raise to SO(3)
    R_bar=T_w_to_c[0:3, 0:3]@R_bar      # orientation in camera frame
    
    R_tag=feature["R"]
    
    tau_bar= Log(R_bar)
    dtau = Log(R_tag) - tau_bar#measurement error 
    dtau = Log(Exp(dtau))
    
    jr=-Jl_inv(tau_bar)@T_c_to_r[0:3,0:3].T@[0,0,1] #jacobian of robot orientation
    jtag=Jr_inv(tau_bar)@[0,0,1]    #jacobian of tag orientation
    
    Jloc=get_pixel_jacobian(mu, xl) #jacobian of robot pose (x,y, theta) and tag location (x,y,z)
    
    H=np.zeros((6,7)) #number of obervation: 6, number of state:7 
    H[0:3, 0:6] = Jloc
    H[3:6, 2] = jr
    H[3:6:, 6] = jtag
    
    F=np.zeros((7,mu.shape[0]))
    F[0:3,0:3]=np.eye(3)
    F[3:7, idx:idx+4]=np.eye(4) 

    H=H@F
    k=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
    dz=feature["t"]-z_bar
    dz=np.concatenate((dz, dtau))
    dmu+=k@(dz)
    sigma=(np.eye(mu.shape[0])-k@H)@(sigma)
    
    dmu[2]=angle_wrapping(dmu[2])
    dmu[idx+3]=angle_wrapping(dmu[idx+3])

# mu[3:]=mu[3:]+dmu[3:]
# mu[0:3]=mu[0:3]+dmu[0:3]
mu=mu+dmu
sigma[0:3, 0:3]=sigma[0:3,0:3]+R

plt.arrow(mu[0], mu[1], 0.1*cos(mu[2]), 0.1*sin(mu[2]))
for i in landmarks.values():
    plt.arrow(mu[i], mu[i+1], 0.1*cos(mu[i+3]), 0.1*sin(mu[i+3]))
plt.axis('scaled')
plt.xlim([0.0, 0.8])
plt.ylim([-0.5, 0.5])
