# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:22:38 2023

@author: hibad
"""

import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
import numpy as np
import os
from numpy import sin, cos, arccos, trace
from numpy.linalg import norm, inv
np.set_printoptions(precision=2)

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

def get_rs_param(cfg):
    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()
    return [intr.fx, intr.fy, intr.ppx, intr.ppy]


os.add_dll_directory(r"C:\Users\hibad\anaconda3\lib\site-packages\pupil_apriltags.libs")
cv2.destroyAllWindows()
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

n=12
mu=np.zeros(n)

sigma=np.eye(n)*0
# sigma=np.zeros((n,n))
# sigma[3:6,3:6]*=0
Q=np.eye(n)
Q[0:3,0:3]*=0.01
Q[3:6,3:6]*=1
Q[6:9,6:9]*=0
Q[9:12,9:12]*=0
tags={}

init=True
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        rgb = np.asanyarray(color_frame.get_data())
        img=np.hstack((rgb, rgb))
        #Tag detection
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        result=at_detector.detect(gray, estimate_tag_pose=True, tag_size=TAG_SIZE, 
        				camera_params=cam_param)
        
        rgb_raw=rgb.copy()
        rgb_kalman=rgb.copy()

        for r in result:
            xp=r.center[0]
            yp=r.center[1] 
            tag_id=r.tag_id
            R=r.pose_R
            R[:, 2]=np.cross( R[:, 0],  R[:, 1])
            R=R@np.array([[0,1,0],
                        [0,0,-1],
                        [-1,0,0]])
            t=r.pose_t

            rgb_raw=draw_frame(rgb_raw,R,t, K)
            tags[r.tag_id]={"R": R, "t":t.flatten() }
            
        if len(result)==2:
            odom=np.eye(4)
            odom[0:3, 0:3]= tags[0]["R"].T
            odom[0:3,3]= tags[0]["R"].T@tags[0]["t"]
            if init:    
                odom_prev=odom
                mu[9:12]=Log(tags[1]['R'])
                init=False
           
            X_bar=np.eye(4)
            X_bar[0:3,0:3] = Exp(mu[3:6])
            X_bar[0:3,3] = mu[0:3]
            # X_bar=X_bar@(inv(odom_prev)@odom)
           
            sigma=sigma+Q
            # mu[3:6]=Log(X_bar[0:3,0:3])
            # mu[0:3] = X_bar[0:3,3]
            # odom_prev=odom

            tau_bar=Log(X_bar[0:3,0:3].T@Exp(mu[9:12]))
            
            tau_tag=Log(tags[1]["R"])
            F=np.zeros((6,12))
            F[3:6,9:12]=np.eye(3)
            F[0:3,3:6]=np.eye(3)
            
            jr=-Jl_inv(tau_bar)@Jr(mu[3:6])
            jtag=Jr_inv(tau_bar)@Jr(mu[9:12])
           # jr=-Jl_inv(tau_bar)
         #   jtag=Jr_inv(tau_bar)
            J=np.concatenate((jr, jtag), 1)
            
            H=J@F
            dr=tau_tag-tau_bar
            gain=sigma@H.T@np.linalg.inv(H@sigma@H.T+np.eye(3)*10)
            dX=gain@dr
            mu=mu+dX
        #    mu[3:6]=Log(X_bar[0:3,0:3]@Exp(dX[3:6]))
         #   mu[9:12]=Log(Exp(mu[9:12])@Exp(dX[9:12]))
         
            
            sigma=(np.eye(mu.shape[0])-gain@H)@(sigma)
            
        rgb_kalman=draw_frame(rgb_kalman,Exp(mu[3:6]).T@Exp(mu[9:12]),tags[1]['t'].reshape(3,1), K)
        # rgb_kalman=draw_frame(rgb_kalman,Exp(mu[3:6]).T,tags[0]['t'].reshape(3,1), K)
        img=np.hstack((rgb_raw, rgb_kalman))
            # image=cv2.circle(image, (int(x[0]),int(x[1])), radius=5, color=[0,0,255], thickness=-1)
        cv2.imshow('Raw', img)

        cv2.waitKey(1)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    pipeline.stop()
                       