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
from numpy import sin, cos
from numpy.linalg import norm, inv
import message_filters
import tf
import time
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
np.float = np.float64 
import ros_numpy
import threading
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

        T_c_to_r, T_r_to_c = get_camera_to_robot_tf()

        self.T_c_to_r=T_c_to_r
        self.T_r_to_c=T_r_to_c
        self.lock=threading.Lock()
        camera_info = self.get_message("/camera/rgb/camera_info", CameraInfo)
        self.K = np.reshape(camera_info.K, (3,3))
        self.K_inv=np.linalg.inv(self.K)
        self.t=time.time()
        self.marker_pub = rospy.Publisher("/apriltags", Marker, queue_size = 2)
        self.image_pub = rospy.Publisher("/camera/rgb/rgb_detected", Image, queue_size = 2)
        
        #motion covariance
        self.R=np.eye(6)
        self.R[0,0]=0.001
        self.R[1,1]=0.001
        self.R[2,2]=0.0001
        self.R[3:5, 3:5] *= 0.001
        self.R[5,5] *= 0.1

        #observation covariance
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

        R=tf.transformations.quaternion_matrix([odom.pose.pose.orientation.x,
                                                   odom.pose.pose.orientation.y,
                                                   odom.pose.pose.orientation.z,
                                                   odom.pose.pose.orientation.w])[0:3,0:3]
        M=np.eye(4)
        M[0:3,0:3] = R
        M[0:3,3]=[odom.pose.pose.position.x,
                  odom.pose.pose.position.y,
                  odom.pose.pose.position.z]
        
        self.odom_prev = M    
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
            self.cloud, depth = msg2pc(pc_msg)
            T =  np.ascontiguousarray(self.K_inv.copy()@self.T_c_to_r[0:3,0:3].copy())
            self.cloud_cov = get_cloud_covariance_par(np.ascontiguousarray(depth),  np.ascontiguousarray(self.Q), T)
            indx=~np.isnan(depth.reshape(-1))
            self.cloud=self.cloud.select_by_index(np.where(indx)[0])
            self.cloud_cov = self.cloud_cov[indx]
            self.cloud.transform(self.T_c_to_r)
            self.id=node_id
            self.mu=[np.eye(4)]
            self.sigma=np.zeros((6,6))
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
        
    def get_message(self, topic, msgtype):
        	try:
        		data=rospy.wait_for_message(topic,msgtype)
        		return data 
        	except rospy.ServiceException as e:
        		print("Service all failed: %s"%e)

    def odom_callback(self, data):
        self.sigma+=np.eye(len(self.sigma))*0.001
        return 
        with self.lock:

            R=tf.transformations.quaternion_matrix([data.pose.pose.orientation.x,
                                                        data.pose.pose.orientation.y,
                                                        data.pose.pose.orientation.z,
                                                        data.pose.pose.orientation.w])[0:3,0:3]

            odom = np.eye(4)
            odom[0:3,0:3] = R
            odom[0:3,3]=[data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z]
            
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
            self.sigma=(Jx)@self.sigma@(Jx.T)+F.T@(Ju)@self.R@(Ju.T)@F
            # self.sigma+=np.eye(len(self.sigma))*0.001
            self.odom_prev=odom
        
    def detect_apriltag(self,rgb, depth):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        result=self.at_detector.detect(gray, estimate_tag_pose=True, tag_size=0.13636, 
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
            if z<2:
                features[r.tag_id]= {"xp": xp, "yp": yp, "z":z, "M":self.T_c_to_r@M }
        return features
    
        

    def _initialize_new_landmarks(self, landmarks):
        mu=self.mu.copy()       #current point estimates 
        sigma=self.sigma.copy() #current covariance
        for landmark_id in landmarks:
            if not landmark_id in self.landmarks.keys():
                landmark=landmarks[landmark_id]
                
                M = mu[0]@landmark["M"].copy() #feature orientation in world frame 
                
                self.landmarks[landmark_id]=len(mu)
                mu.append(M)
                sigma_new=np.diag(np.ones(sigma.shape[0]+6)*99999999999)
                sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
                sigma=sigma_new
                
        self.sigma=sigma
        self.mu=mu

    def _correction(self,features):
        mu=self.mu.copy()
        sigma=self.sigma.copy()
        
        n = len(features)
        H=np.zeros((6*n,6*len(mu)))
        Q=np.zeros((6*n,6*n))
        dz = np.zeros(6*n)
        
        for i,feature_id in enumerate(features):    
            feature=features[feature_id]
            idx=self.landmarks[feature_id]
            
            #global feature location
            M_tag_bar = mu[idx].copy() 
            Z_bar = mu[0]@M_tag_bar  #feature location in camera frame
            z_bar = SE3.Log(Z_bar)
      
            Z = feature["M"]
            z = SE3.Log(Z)
            
            dz[6*i:6*i+6] = SE3.Log(SE3.Exp(z - z_bar)) #measurement error 

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
            J_z = SE3.Jr_inv(z)
            # Q[6*i:6*i+6, 6*i:6*i+6] =J_z@self.Q.copy()@J_z.T
            Q[6*i:6*i+6, 6*i:6*i+6] =self.Q.copy()
            
        K=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
        sigma=(np.eye(len(mu)*6)-K@H)@(sigma)
        dmu=K@(dz)
        # for i in range(len(mu)):
        #     self.mu[i]=mu[i]@SE3.Exp(dmu[6*i:6*i+6])
        
        # self.mu[0]=mu[0]@SE3.Exp(-dmu[0:6])
        self.mu[0]=SE3.Exp(SE3.Log(mu[0])+ dmu[0:6])
        self.sigma=(sigma+sigma.T)/2
        
    def camera_callback(self, rgb_msg, depth_msg):
        with self.lock:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg,"bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg,"32FC1")
            features=self.detect_apriltag(rgb, depth)
            for feature in features.values():
                rgb=draw_frame(rgb, feature, self.K)
            self._initialize_new_landmarks(features)
            self._correction(features)
            #self.image_pub.publish(self.bridge.cv2_to_imgmsg(rgb))
            
def get_pose_marker(tags, mu):
    markers=[]
    for tag_id, idx in tags.items():
        marker=Marker()
        M=mu[idx]
        p=Pose()
        p.position.x = M[0,3]
        p.position.y = M[1,3]
        p.position.z = M[2,3]
        q=tf.transformations.quaternion_from_matrix(M)
        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]

    
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
        M = ekf.mu[0]
        br.sendTransform((M[0,3], M[1,3] , M[2,3]),
                        tf.transformations.quaternion_from_matrix(M),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
  

        rate.sleep()
