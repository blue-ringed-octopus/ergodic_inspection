#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:40:13 2023

@author: barc
"""
import rospy 
import rospkg
import yaml
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
from numpy.linalg import inv
import message_filters
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose 
np.float = np.float64 
import ros_numpy
import threading
from apriltag_EKF_SE3 import EKF

np.set_printoptions(precision=2)


class EKF_Wrapper:
    def __init__(self, node_id, tf_br, params , landmarks={}, fixed_landmarks=[-1]):
        self.params = params
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf_br
        self.bridge = CvBridge()

        T_c_to_r = self.get_camera_to_robot_tf()
        self.lock=threading.Lock()
        camera_info = self.get_message(params["EKF"]["camera_info"], CameraInfo)

        K = np.reshape(camera_info.K, (3,3))
        print("received camera info: ", K)

        self.marker_pub = rospy.Publisher(params["EKF"]["apriltag_marker_topic"], Marker, queue_size = 2)
        self.image_pub = rospy.Publisher(params["EKF"]["rgb_detected"], Image, queue_size = 2)
        
       
        odom=rospy.wait_for_message(params["EKF"]["odom_topic"],Odometry)
        R=tf.transformations.quaternion_matrix([odom.pose.pose.orientation.x,
                                                   odom.pose.pose.orientation.y,
                                                   odom.pose.pose.orientation.z,
                                                   odom.pose.pose.orientation.w])[0:3,0:3]
        print("received initial odometry")

        M=np.eye(4)
        M[0:3,0:3] = R
        M[0:3,3]=[odom.pose.pose.position.x,
                  odom.pose.pose.position.y,
                  odom.pose.pose.position.z]
        
        self.ekf = EKF(node_id=node_id, 
                       T_c_to_r = T_c_to_r, 
                       K = K, 
                       odom = M, 
                       R = np.array(params["EKF"]["motion_noise"]),
                       Q = np.array(params["EKF"]["observation_noise"]),
                       tag_size = params["EKF"]["tag_size"],
                       tag_family = params["EKF"]["tag_families"],
                       fixed_landmarks = fixed_landmarks)
        self.reset(node_id, landmarks)

        rospy.Subscriber(params["EKF"]["odom_topic"], Odometry, self.odom_callback)
        rgbsub=message_filters.Subscriber(params["EKF"]["rgb_topic"], Image)
        depthsub=message_filters.Subscriber(params["EKF"]["depth_aligned_topic"], Image)

        ts = message_filters.ApproximateTimeSynchronizer([rgbsub, depthsub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.camera_callback)

    def get_camera_to_robot_tf(self):
        self.tf_listener.waitForTransform(self.params["EKF"]["robot_frame"],self.params["EKF"]["optical_frame"],rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.tf_listener.lookupTransform(self.params["EKF"]["robot_frame"], self.params["EKF"]["optical_frame"], rospy.Time(0))
        T_c_to_r = self.tf_listener.fromTranslationRotation(trans, rot)
        return T_c_to_r
    
    def reset(self, node_id, landmarks={}, get_point_cloud = True, fixed_landmarks=None):
        self.recent_features={}
        print("reseting EKF")
        with self.lock:
            pc_info = None
            if get_point_cloud:
                pc_info = self.get_point_cloud()
            self.id = node_id
            self.ekf.reset(node_id, pc_info, landmarks, fixed_landmarks = fixed_landmarks)
        print("EKF initialized") 
        
    def get_point_cloud(self):
        pc_msg=rospy.wait_for_message(self.params["EKF"]["depth_pointcloud_topic"],PointCloud2)
        pc_info = self.msg2pc(pc_msg)
        return pc_info
    
    def msg2pc(self, msg):
        pc=ros_numpy.numpify(msg)
        m,n = pc['x'].shape
        depth = pc['z']
        x=pc['x'].reshape(-1)
        points=np.zeros((len(x),3))
        points[:,0]=x
        points[:,1]=pc['y'].reshape(-1)
        points[:,2]=pc['z'].reshape(-1)
        pc=ros_numpy.point_cloud2.split_rgb_field(pc)
        img = np.zeros((m,n,3))
        img[:,:,0] = pc['r']
        img[:,:,1] = pc['g']
        img[:,:,2] = pc['b']


        rgb=np.zeros((len(x),3))
        rgb[:,0]=pc['r'].reshape(-1)
        rgb[:,1]=pc['g'].reshape(-1)
        rgb[:,2]=pc['b'].reshape(-1)

        p = {"points": points, "colors": np.asarray(rgb/255)}
        return p, depth, img.astype('uint8')    
    
    def get_message(self, topic, msgtype):
        	try:
        		data=rospy.wait_for_message(topic,msgtype)
        		return data 
        	except rospy.ServiceException as e:
        		print("Service all failed: %s"%e)

    def odom_callback(self, data):
            R=tf.transformations.quaternion_matrix([data.pose.pose.orientation.x,
                                                        data.pose.pose.orientation.y,
                                                        data.pose.pose.orientation.z,
                                                        data.pose.pose.orientation.w])[0:3,0:3]

            odom = np.eye(4)
            odom[0:3,0:3] = R
            odom[0:3,3]=-[data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z]
            
            Rv = np.eye(6)
            Rv[0,0] = data.twist.twist.linear.x**2
            Rv[1,1] = data.twist.twist.linear.y**2
            Rv[2,2] =  data.twist.twist.linear.z**2 
            Rv[3,3] =  data.twist.twist.angular.x**2 
            Rv[4,4] =  data.twist.twist.angular.y**2
            Rv[5,5] =  5 * data.twist.twist.angular.z**2
            with self.lock:
                self.ekf.motion_update(odom.copy(), Rv)
                
            self.tf_listener.waitForTransform(self.params["EKF"]["odom_frame"],self.params["EKF"]["robot_frame"],rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(self.params["EKF"]["odom_frame"],self.params["EKF"]["robot_frame"], rospy.Time(0))
            odom = self.tf_listener.fromTranslationRotation(trans, rot)
            M = self.ekf.mu[0].copy()
            M = M@inv(odom)
            self.tf_br.sendTransform((M[0,3], M[1,3] , M[2,3]),
                            tf.transformations.quaternion_from_matrix(M),
                            rospy.Time.now(),
                            self.params["EKF"]["odom_frame"],
                            "ekf")
            
    def camera_callback(self, rgb_msg, depth_msg):
        with self.lock:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg,"bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg,"32FC1")
            features_obsv = self.ekf.camera_update(rgb, depth)
        self.recent_features = features_obsv
    def tf_callback(self, msg):
        pass
    def get_recent_features(self):
        return self.recent_features.copy()
    
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

def pc_to_msg(pc):
    points = pc["points"]
    colors = pc["colors"]
    pc_array = np.zeros(len(points), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('r', np.uint32),
    ('g', np.uint32),
    ('b', np.uint32),
    ])

    pc_array['x'] = points[:,0]
    pc_array['y'] = points[:, 1]
    pc_array['z'] = points[:, 2]
    pc_array['r'] = (colors[:,0]*255).astype(np.uint32)
    pc_array['g'] = (colors[:, 1]*255).astype(np.uint32)
    pc_array['b'] = (colors[:, 2]*255).astype(np.uint32)
    pc_array= ros_numpy.point_cloud2.merge_rgb_fields(pc_array)
    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=rospy.Time.now(), frame_id="map")
    
    return pc_msg

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    remap_tag_ids = rospy.get_param("remapTags")
    remap_tag_ids = remap_tag_ids.split(',')
    remap_tag_ids = [int(id_) for id_ in remap_tag_ids]
    
    rospack=rospkg.RosPack()
    path = rospack.get_path("ergodic_inspection")
    is_sim = rospy.get_param("isSim")
    
    if is_sim:
        param_path = path + "/param/sim/"
        resource_path = path + "/resources/sim/"
    else:
        param_path = path +"/param/real/"
        resource_path = path + "/resources/real/"

    with open(param_path+'estimation_param.yaml', 'r') as file:
        params = yaml.safe_load(file)
        
    with open(resource_path+'prior_features.yaml', 'r') as file:
        prior =  yaml.safe_load(file)
    
    landmarks = {}
    fixed_landmarks=[]
    for id_, tag in prior.items():
        M = np.eye(4)
        M[0:3,0:3] = R.from_euler('xyz', tag["orientation"]).as_matrix()
        M[0:3,3] =  tag["position"]
        landmarks[id_] = M
        if not remap_tag_ids[0] == -1:
            if (id_ not in remap_tag_ids):
                fixed_landmarks.append(id_)
        
    rospy.init_node('EKF',anonymous=False)
    pc_pub=rospy.Publisher("/pc_rgb", PointCloud2, queue_size = 2)
    factor_graph_marker_pub = rospy.Publisher("/factor_graph", MarkerArray, queue_size = 2)
    br = tf.TransformBroadcaster()

    wrapper = EKF_Wrapper(0, br, params, landmarks=landmarks, fixed_landmarks = fixed_landmarks)
    rate = rospy.Rate(30) # 10hz
    while not rospy.is_shutdown():
        pc = wrapper.ekf.cloud["pc"]
        pc_pub.publish(pc_to_msg(pc))
        markers=get_pose_marker(wrapper.ekf.features, wrapper.ekf.mu)
        factor_graph_marker_pub.publish(markers)
        br.sendTransform((0,0 , 0),
                        tf.transformations.quaternion_from_matrix(np.eye(4)),
                        rospy.Time.now(),
                        "ekf",
                        "map")

        rate.sleep()
        
    print("saving features to prior")

    for tag_id, idx in wrapper.ekf.features.items():
            T = wrapper.ekf.mu[idx]
            rot = R.from_matrix(T[0:3, 0:3]).as_euler("xyz") 
            t = T[0:3, 3]
            prior[tag_id] = {"position": t.tolist(), "orientation": rot.tolist()}    

            
    with open('tag_loc.yaml', 'w') as file:
        yaml.safe_dump(prior, file)