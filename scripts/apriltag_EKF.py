#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:40:13 2023

@author: barc
"""
import rospy 
import rospkg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from pupil_apriltags import Detector
import numpy as np
from numpy import sin, cos, arccos, trace
from numpy.linalg import norm, inv
import message_filters
import tf
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
np.float = np.float64 
import ros_numpy
import threading
from common_functions import angle_wrapping, v2t, t2v
import open3d as o3d 
from numba import cuda
from Lie import SO3, SE3, SE2, SO2
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

rospack=rospkg.RosPack()
np.set_printoptions(precision=2)

def get_camera_to_robot_tf():
    listener=tf.TransformListener()
    listener.waitForTransform('/base_footprint','/camera_rgb_optical_frame',rospy.Time(), rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform('/base_footprint', '/camera_rgb_optical_frame', rospy.Time(0))
    T_c_to_r=listener.fromTranslationRotation(trans, rot)
    T_r_to_c=np.linalg.inv(T_c_to_r)
    return T_c_to_r, T_r_to_c

def msg2pc(msg):
    pc=ros_numpy.numpify(msg)
    m,n = pc['x'].shape
    depth = pc['z']
    x=pc['x'].reshape(-1)
    points=np.zeros((len(x),3))
    points[:,0]=x
    points[:,1]=pc['y'].reshape(-1)
    points[:,2]=pc['z'].reshape(-1)
    pc=ros_numpy.point_cloud2.split_rgb_field(pc)
    rgb=np.zeros((len(x),3))
    rgb[:,0]=pc['r'].reshape(-1)
    rgb[:,1]=pc['g'].reshape(-1)
    rgb[:,2]=pc['b'].reshape(-1)
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(points)
    p.colors=o3d.utility.Vector3dVector(np.asarray(rgb/255))
    
    return p, depth   

def draw_frame(img, tag, K):
    img=cv2.circle(img, (int(tag["xp"]), int(tag["yp"])), 5, (0, 0, 255), -1)
    M=tag["M"].copy()
    
    x_axis=K@M[0:3,:]@np.array([0.06,0,0,1])
    x_axis=x_axis/(x_axis[2])
    
    img=cv2.arrowedLine(img, (int(tag["xp"]), int(tag["yp"])), (int(x_axis[0]), int(x_axis[1])), 
                                     (0,0,255), 5)  
    return img


class EKF:
    def __init__(self, node_id):
        self.bridge = CvBridge()

        self.fr = np.zeros((6,3))
        self.fr[0,0]=1
        self.fr[1,1]=1
        self.fr[5,2]=1
        self.ftag = np.zeros((6,4))
        self.ftag[0,0]=1
        self.ftag[1,1]=1
        self.ftag[2,2] = 1
        self.ftag[5,3] = 1
        T_c_to_r, T_r_to_c = get_camera_to_robot_tf()

        self.T_c_to_r=T_c_to_r
        self.T_r_to_c=T_r_to_c
        self.lock=threading.Lock()
        camera_info = self.get_message("/camera/rgb/camera_info", CameraInfo)
        self.K = np.reshape(camera_info.K, (3,3))
        self.K_inv=np.linalg.inv(self.K)
        self.mu=np.zeros(3)
        self.t=time.time()
        self.marker_pub = rospy.Publisher("/apriltags", Marker, queue_size = 2)
        self.image_pub = rospy.Publisher("/camera/rgb/rgb_detected", Image, queue_size = 2)

        self.R=np.eye(3)
        self.R[0,0]=0.01
        self.R[1,1]=0.01
        self.R[2,2]=0.1
        
        self.Q=np.eye(6)
        self.Q[0,0]=1**2 # 
        self.Q[1,1]=1**2 # 
        self.Q[2,2]=1**2 #
        self.Q[3:6, 3:6] *= (np.pi/2)**2 #axis angle
        
        
        self.at_detector = Detector(
                    families="tag36h11",
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                    )
        odom=rospy.wait_for_message("/odom",Odometry)

        M=tf.transformations.quaternion_matrix([odom.pose.pose.orientation.x,
                                                   odom.pose.pose.orientation.y,
                                                   odom.pose.pose.orientation.z,
                                                   odom.pose.pose.orientation.w])
        theta = SO3.Log(M[0:3,0:3])
        R = SO2.Exp(theta[2])
        self.odom_prev = np.eye(3)
        self.odom_prev[0:2,0:2] = R
        self.odom_prev[0:2,2]=[odom.pose.pose.position.x,
                          odom.pose.pose.position.y]
    
        self.reset(node_id)


        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        
        rgbsub=message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
        depthsub=message_filters.Subscriber("/camera/depth_registered/image_raw", Image)

        ts = message_filters.ApproximateTimeSynchronizer([rgbsub, depthsub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.camera_callback)

        
    def reset(self, node_id):
        print("reseting EKF")
        with self.lock:
            pc_msg=rospy.wait_for_message("/depth_registered/points",PointCloud2)
        #    depth_msg=rospy.wait_for_message("/camera/depth_registered/image_raw", Image)
            self.cloud, depth = msg2pc(pc_msg)
           # depth=self.bridge.imgmsg_to_cv2(depth_msg,"32FC1")
            T =  np.ascontiguousarray(self.K_inv.copy()@self.T_c_to_r[0:3,0:3].copy())
            self.cloud_cov = get_cloud_covariance_par(np.ascontiguousarray(depth),  np.ascontiguousarray(self.Q), T)
            indx=~np.isnan(depth.reshape(-1))
            self.cloud=self.cloud.select_by_index(np.where(indx)[0])
            self.cloud_cov = self.cloud_cov[indx]
            self.cloud.transform(self.T_c_to_r)


            self.id=node_id
            self.mu=np.zeros(3)
            self.sigma=np.zeros((3,3))
            self.landmarks={}

        print("EKF initialized")
        
    
    # def get_cloud_covariance(self, depth):
    #     n, m = depth.shape
    #     T=self.T_c_to_r[0:3,0:3].copy()@inv(self.K.copy())
    #     J=[T@np.array([[depth[i,j],0,i],
    #                 [0,depth[i,j],j],
    #                 [0,0,1]]) for i in range(n) for j in range(m)]
    
    #     cov=np.asarray([j@self.Q[0:3,0:3]@j for j in J])
    #     return cov
    
    def get_tf(self):
        mu=self.mu[0:3].copy()
        return v2t([mu[0], mu[1], 0 ,mu[2]])
        
    
        
    def get_message(self, topic, msgtype):
        	try:
        		data=rospy.wait_for_message(topic,msgtype)
        		return data 
        	except rospy.ServiceException as e:
        		print("Service all failed: %s"%e)

    def odom_callback(self, data):
        with self.lock:

            M=tf.transformations.quaternion_matrix([data.pose.pose.orientation.x,
                                                       data.pose.pose.orientation.y,
                                                       data.pose.pose.orientation.z,
                                                       data.pose.pose.orientation.w])
            theta = SO3.Log(M[0:3,0:3])
            R = SO2.Exp(theta[2])
            odom = np.eye(3)
            odom[0:2,0:2] = R
            odom[0:2,2]=[data.pose.pose.position.x,
                              data.pose.pose.position.y]
            
            #get relative transformation
            U = np.linalg.inv(self.odom_prev)@odom
            u = SE2.Log(U)
            
            mu=self.mu.copy()
            tau_prev=mu[0:3]
            tau =SE2.Log(SE2.Exp(tau_prev)@U)
            mu[0:3] = tau
            
            F=np.zeros((3,mu.shape[0]))
            F[0:3,0:3]=np.eye(3)
            
            
            Jx=SE2.Jr_inv(tau)@inv(SE2.Ad(U))@SE2.Jr(tau_prev)
            
            Jx = F.T@Jx@F
            Ju=SE2.Jr_inv(tau)@SE2.Jr(u)
            self.mu = mu
            self.sigma=(Jx)@self.sigma@(Jx.T)+F.T@(Ju)@self.R@(Ju.T)@F
            self.odom_prev=odom
        
    def detect_apriltag(self,rgb, depth):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        result=self.at_detector.detect(gray, estimate_tag_pose=True, tag_size=0.13636, 
        				camera_params=[self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        landmarks={}
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
            if z<2:
                landmarks[r.tag_id]= {"xp": xp, "yp": yp, "z":z, "M":M }
        return landmarks
    
        

    def _initialize_new_landmarks(self, landmarks):
        mu=self.mu.copy()       #current point estimates 
        sigma=self.sigma.copy() #current covariance
        T=SE3.Exp([mu[0], mu[1], 0,0,0, mu[2]])@self.T_c_to_r    #coordinate transformation from camera coordinate to world coordinate
        for landmark_id in landmarks:
            if not landmark_id in self.landmarks.keys():
                landmark=landmarks[landmark_id]
                
                M = T@landmark["M"].copy()#feature orientation in world frame 
                
                #remove x,y rotation
                R=M[0:3,0:3] #feature orientation in camera frame
                theta=SO3.Log(R)
                theta[0:2]=[0,0]
                R=SO3.Exp(theta)
                
                M[0:3,0:3]=R

                tau=SE3.Log(M) #tangent space
                tau_hat = self.ftag.T@tau 
                
                self.landmarks[landmark_id]=mu.shape[0]
                mu=np.hstack((mu.copy(),tau_hat))
                sigma_new=np.diag(np.ones(sigma.shape[0]+len(tau_hat))*99999999999)
                sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
                sigma=sigma_new
                
        self.sigma=sigma
        self.mu=mu

    def _correction(self,features):
        mu=self.mu.copy()
        sigma=self.sigma.copy()
        tau_r = np.array([mu[0], mu[1], 0, 0, 0, mu[2]])
        T_c_to_w=SE3.Exp(tau_r)@self.T_c_to_r
        T_w_to_c=inv(T_c_to_w)
        dmu=np.zeros(mu.shape)
        for feature_id in features:    
            feature=features[feature_id]
            idx=self.landmarks[feature_id]
            
            #global feature location
            tau_tag_hat=mu[idx:idx+4].copy() 
            tau_tag_bar = self.ftag@tau_tag_hat
            M_tag_bar = SE3.Exp(tau_tag_bar)
            M_tag_c_bar=T_w_to_c@M_tag_bar  #feature location in camera frame
            tau_tag_c_bar = SE3.Log(M_tag_c_bar)
      
            M_tag_c=feature["M"]

            dtau = SE3.Log(M_tag_c) - tau_tag_c_bar#measurement error 
            dtau = SE3.Log(SE3.Exp(dtau))
            
            J_cr=np.zeros((6,6))
            J_cr[0:3,0:3] = self.T_c_to_r[0:3,0:3].T
            J_cr[3:6,3:6] = self.T_c_to_r[0:3,0:3].T
            J_cr[0:3,3:6] = -self.T_c_to_r[0:3,0:3].T@SO3.hat(self.T_c_to_r[0:3,3])
            
            
            Jr=-SE3.Jl_inv(tau_tag_c_bar)@J_cr@SE3.Jr(tau_r)@self.fr #jacobian of robot pose
            Jtag=SE3.Jr_inv(tau_tag_c_bar)@SE3.Jr(tau_tag_bar)@self.ftag   #jacobian of tag pose
            
            H=np.zeros((6,7)) #number of obervation: 6, number of state:7 
            H[0:6, 0:3] = Jr
            H[0:6:, 3:7] = Jtag
            
            F=np.zeros((7,mu.shape[0]))
            F[0:3,0:3]=np.eye(3)
            F[3:7, idx:idx+4]=np.eye(4) 

            H=H@F
            Q=self.Q.copy()
            K=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
            dmu+=K@(dtau)
            sigma=(np.eye(mu.shape[0])-K@H)@(sigma)

        self.mu=mu+dmu
        self.sigma=sigma
        
    def camera_callback(self, rgb_msg, depth_msg):
        with self.lock:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg,"bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg,"32FC1")
            features=self.detect_apriltag(rgb, depth)
            for feature in features.values():
                rgb=draw_frame(rgb, feature, self.K)
            self._initialize_new_landmarks(features)
            self._correction(features)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb))
            
def get_pose_marker(tags, mu):
    markers=[]
    for tag_id, idx in tags.items():
        marker=Marker()
        x=mu[idx:idx+4]
        M = SE2.Exp([x[0], x[1], x[3]])
        p=Pose()
        p.position.x = M[0,2]
        p.position.y = M[1,2]
        p.position.z = x[2]
        
        p.orientation.w = cos(x[3]/2)
        p.orientation.x = 0
        p.orientation.y = 0
        p.orientation.z = sin(x[3]/2)

    
        marker = Marker()
        marker.type = 0
        marker.id = tag_id
        
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        
        marker.pose.orientation.x=0
        marker.pose.orientation.y=0
        marker.pose.orientation.z=0
        marker.pose.orientation.w=1
        
        
        marker.scale.x = 0.5
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.pose = p
        markers.append(marker)
    markerArray=MarkerArray()
    markerArray.markers=markers
    return markerArray

if __name__ == "__main__":

    rospy.init_node('EKF',anonymous=False)
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)

    ekf=EKF(0)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(30) # 10hz
    while not rospy.is_shutdown():
        # pc_pub.publish(ekf.cloud)
        markers=get_pose_marker(ekf.landmarks, ekf.mu)
        factor_graph_marker_pub.publish(markers)
        M = SE2.Exp(ekf.mu[0:3])
        br.sendTransform((M[0,2], M[1,2] , 0),
                        tf.transformations.quaternion_from_euler(0, 0, ekf.mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
  

        rate.sleep()
