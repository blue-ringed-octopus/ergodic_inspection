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
from geometry_msgs.msg import Point, Pose
np.float = np.float64 
import ros_numpy
import threading
from common_functions import angle_wrapping, v2t, t2v
import open3d as o3d 

rospack=rospkg.RosPack()
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

def get_camera_to_robot_tf():
    listener=tf.TransformListener()
    listener.waitForTransform('/base_footprint','/camera_rgb_optical_frame',rospy.Time(), rospy.Duration(4.0))
    (trans, rot) = listener.lookupTransform('/base_footprint', '/camera_rgb_optical_frame', rospy.Time(0))
    T_c_to_r=listener.fromTranslationRotation(trans, rot)
    T_r_to_c=np.linalg.inv(T_c_to_r)
    return T_c_to_r, T_r_to_c

def msg2pc(msg):
    pc=ros_numpy.numpify(msg)
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
    return p   

def draw_frame(img, tag, K):
    img=cv2.circle(img, (int(tag["xp"]), int(tag["yp"])), 5, (0, 0, 255), -1)
    R=tag["R"]
    t=tag['t']
    
    x_axis=K@np.concatenate((R,t),1)@np.array([0.06,0,0,1])
    x_axis=x_axis/(x_axis[2])
    
    # z_axis=K@np.concatenate((R,t),1)@np.array([0,0,0.06,1])
    # z_axis=z_axis/(z_axis[2])
    
    img=cv2.arrowedLine(img, (int(tag["xp"]), int(tag["yp"])), (int(x_axis[0]), int(x_axis[1])), 
                                     (0,0,255), 5)  
    return img
class EKF:
    def __init__(self, node_id):
        print("EKF initialize")
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
        self.image_pub = rospy.Publisher("//camera/rgb/rgb_detected", Image, queue_size = 2)

        self.R=np.eye(3)
        self.R[0,0]=9999#0.01
        self.R[1,1]=9999#0.01
        self.R[2,2]=9999#0.1
        
        self.Q=np.eye(6)
        self.Q[0,0]=20**2 # x pixel
        self.Q[1,1]=20**2 # y pixel
        self.Q[2,2]=1**2  # depth
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
        self.odom_prev=v2t([odom.pose.pose.position.x,
                          odom.pose.pose.position.y,
                          0,
                          odom.pose.pose.orientation.z])
        self.odom_prev=tf.transformations.quaternion_matrix([odom.pose.pose.orientation.x,
                                                   odom.pose.pose.orientation.y,
                                                   odom.pose.pose.orientation.z,
                                                   odom.pose.pose.orientation.w])
        self.odom_prev[0:3,3]=[odom.pose.pose.position.x,
                          odom.pose.pose.position.y,0]
    
        self.reset(node_id)


        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        
        rgbsub=message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
        depthsub=message_filters.Subscriber("/camera/depth_registered/image_raw", Image)

        ts = message_filters.ApproximateTimeSynchronizer([rgbsub, depthsub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.camera_callback)

        
    def reset(self, node_id):
        with self.lock:
            self.cloud=msg2pc(rospy.wait_for_message("/depth_registered/points",PointCloud2))
            self.cloud.transform(self.T_c_to_r)
            
            self.id=node_id
            self.mu=np.zeros(3)
            self.sigma=np.zeros((3,3))
            self.landmarks={}
            # self.cloud.header.frame_id="node_"+str(node_id)+"_camera"
        print("EKF initialized")
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
            odom=tf.transformations.quaternion_matrix([data.pose.pose.orientation.x,
                                                       data.pose.pose.orientation.y,
                                                       data.pose.pose.orientation.z,
                                                       data.pose.pose.orientation.w])
            odom[0:3,3]=[data.pose.pose.position.x,
                              data.pose.pose.position.y,0]
            
            dX=np.linalg.inv(self.odom_prev)@odom
            
            mu=self.mu.copy()
            mu_r =t2v(v2t([mu[0], mu[1], 0, mu[2]])@dX)
            mu[0:3] = [mu_r[0], mu_r[1], mu_r[3]]
            F=np.zeros((3,mu.shape[0]))
            F[0:3,0:3]=np.eye(3)
            fx = np.array([[0, 0, - dX[1,3]*cos(mu[2]) - dX[0,3]*sin(mu[2])],
                           [0, 0,   dX[0,3]*cos(mu[2]) - dX[1,3]*sin(mu[2])],
                           [0, 0,                               0]]) 
            
            fx=np.eye(mu.shape[0])+F.T@fx@F
            fu=np.array([[cos(mu[2]), -sin(mu[2]), 0],
                         [sin(mu[2]),  cos(mu[2]), 0],
                         [         0,           0, 1]   ])    
            
            self.mu = mu
            self.sigma=(fx)@self.sigma@(fx.T)+F.T@(fu)@self.R@(fu.T)@F
            self.odom_prev=odom
        
    def detect_apriltag(self,rgb, depth):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        result=self.at_detector.detect(gray, estimate_tag_pose=True, tag_size=0.13636, 
        				camera_params=[self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        landmarks={}
        for r in result:
            xp=r.center[0]
            yp=r.center[1] 
            tag_id=r.tag_id
            rgb=cv2.circle(rgb, (int(xp), int(yp)), 5, (0, 0, 255), -1)
            z=depth[int(yp), int(xp)]
            z=r.pose_t.flatten()[2]
            R=r.pose_R
            R[:, 2]=np.cross(R[:, 0], R[:, 1])
            
            R=R@np.array([[0,1,0],
                            [0,0,-1],
                            [-1,0,0]]) #rotate such that x-axis points outward, z-axis points upward 
            if not np.isnan(z):
                landmarks[tag_id]= {"xp": xp, "yp": yp, "z":z, "R": R, "t": r.pose_t}
        return landmarks
    
        

    def _initialize_new_landmarks(self, landmarks):
        mu=self.mu.copy()       #current point estimates 
        sigma=self.sigma.copy() #current covariance
        T=v2t([mu[0], mu[1], 0, mu[2]])@self.T_c_to_r    #coordinate transformation from camera coordinate to world coordinate
        for landmark_id in landmarks:
            if not landmark_id in self.landmarks.keys():
                landmark=landmarks[landmark_id]
                loc=landmark['z']*self.K_inv@np.array([landmark["xp"], landmark["yp"],1])  
                loc=T@np.hstack((loc,[1]))           
                loc=loc[0:3]
                
                R=landmark["R"] #feature orientation in camera frame
                
                
                
                R=T[0:3,0:3]@R  #feature orientation in world frame 
                
                tau=Log(R) #axis-angle representation 
                
                zeta=np.hstack((loc, tau[2])) #only take the z rotation
                self.landmarks[landmark_id]=mu.shape[0]
                mu=np.hstack((mu.copy(),zeta))
                sigma_new=np.diag(np.ones(sigma.shape[0]+len(zeta))*99999999999)
                sigma_new[0:sigma.shape[0], 0:sigma.shape[0]]=sigma.copy()
                sigma=sigma_new
                
        self.sigma=sigma
        self.mu=mu
        
    def get_pixel_jacobian(self,mu, xl, kx):
        c=cos(mu[2])
        s=sin(mu[2])
        
        jw=np.array([[-c, -s, c*(xl[1]-mu[1]) + s*(mu[0]-xl[0]) , c  , s , 0],
                     [s , -c, c*(mu[0]-xl[0]) + s*(mu[1]-xl[1]) , -s , c , 0],
                     [0 , 0 , 0                                 , 0  , 0 , 1]
                     ])
        
        jr=self.T_r_to_c[0:3, 0:3]
        jc=np.array([[1/kx[2],0 ,-kx[0]/kx[2]**2],
                     [0,1/kx[2], -kx[1]/kx[2]**2],
                     [0,0,1]
                     ])@self.K
        
        return jc@jr@jw
        
    def _correction(self,features):
        mu=self.mu.copy()
        sigma=self.sigma.copy()
        
        T_c_to_w=v2t([mu[0], mu[1], 0, mu[2]])@self.T_c_to_r
        T_w_to_c=inv(T_c_to_w)
        dmu=np.zeros(mu.shape)
        for feature_id in features:    
            feature=features[feature_id]
            idx=self.landmarks[feature_id]
            
            xl=mu[idx:idx+3].copy() #global feature location
            
            x_camera=T_w_to_c@np.concatenate((xl, [1])) #feature location in camera frame
            kx=self.K@x_camera[0:3]                     
            z_bar=np.array([[1/x_camera[2], 0,0],
                            [0,1/x_camera[2],0],
                            [0,0,1]])@kx                #feature on image plane and depth
                        
            theta=angle_wrapping(mu[idx+3].copy()) #estimated planar orientation of the tag
            
            R_bar=Exp([0,0,theta])        #raise to SO(3)
            R_bar=T_w_to_c[0:3, 0:3]@R_bar      # orientation in camera frame
            
            R_tag=feature["R"]
            
            tau_bar= Log(R_bar)
            dtau = Log(R_tag) - tau_bar#measurement error 
            
            jr=-Jl_inv(tau_bar)@self.T_r_to_c[0:3,0:3]@[0,0,1] #jacobian of robot orientation
            jtag=Jr_inv(tau_bar)@[0,0,1]    #jacobian of tag orientation
            
            Jloc=self.get_pixel_jacobian(mu, xl, kx) #jacobian of robot pose (x,y, theta) and tag location (x,y,z)
            
            H=np.zeros((6,7)) #number of obervation: 6, number of state:7 
            H[0:3, 0:6] = Jloc
            H[3:6, 2] = jr
            H[3:6:, 6] = jtag
            
            F=np.zeros((7,mu.shape[0]))
            F[0:3,0:3]=np.eye(3)
            F[3:7, idx:idx+4]=np.eye(4) 

            H=H@F
            Q=self.Q.copy()
        #    Q[0:3, 0:3]=inv(Jc)@Q[0:3, 0:3]@inv(Jc).T
            K=sigma@(H.T)@inv((H@sigma@(H.T)+Q))
            dz=np.array([feature["xp"], feature['yp'], feature['z']])-z_bar
         #   dz=feature["t"].flatten()-x_camera[0:3]
            dz=np.concatenate((dz, dtau))
           # mu = mu + K@(dz)
            dmu+=K@(dz)
            sigma=(np.eye(mu.shape[0])-K@H)@(sigma)
            
            dmu[2]=angle_wrapping(dmu[2])
            dmu[idx+3]=angle_wrapping(dmu[idx+3])

        

        self.mu=mu+dmu
        for idx  in self.landmarks.values():
            self.mu[idx+3]=angle_wrapping(self.mu[idx+3])
        self.sigma=sigma
        
    def camera_callback(self, rgb_msg, depth_msg):
        with self.lock:
            bridge = CvBridge()
            rgb = bridge.imgmsg_to_cv2(rgb_msg,"bgr8")
            depth = bridge.imgmsg_to_cv2(depth_msg,"32FC1")
            features=self.detect_apriltag(rgb, depth)
            for feature in features.values():
                rgb=draw_frame(rgb, feature, self.K)
            self._initialize_new_landmarks(features)
            self._correction(features)
            self.image_pub.publish(bridge.cv2_to_imgmsg(rgb))
            
def get_pose_marker(tags, mu):
    markers=[]
    for tag_id, idx in tags.items():
        marker=Marker()
        x=mu[idx:idx+4]
        p=Pose()
        p.position.x=x[0]
        p.position.y=x[1]
        p.position.z=x[2]
        
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
        br.sendTransform((ekf.mu[0], ekf.mu[1] , 0),
                        tf.transformations.quaternion_from_euler(0, 0, ekf.mu[2]),
                        rospy.Time.now(),
                        "base_footprint",
                        "map")
  

        rate.sleep()
