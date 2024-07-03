#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:04:29 2024

@author: hibad
"""

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseArray, Pose
import ros_numpy
import numpy as np
from ergodic_inspection.srv import PointCloudWithEntropy
import open3d as o3d
def msg2pc(msg):
    pc=ros_numpy.numpify(msg)
    m,n = pc['x'].shape
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
    h = pc["h"]
    print(h)
    # p=o3d.geometry.PointCloud()
    # p.points=o3d.utility.Vector3dVector(points)
    # p.colors=o3d.utility.Vector3dVector(np.asarray(rgb/255))
    p = {"points": points, "colors": np.asarray(rgb/255)}
    return p
def simple_move(x,y,w,z):


    sac = actionlib.SimpleActionClient('move_base', MoveBaseAction )

    #create goal
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = w
    goal.target_pose.pose.orientation.z = z
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()

    #start listner
    sac.wait_for_server()
    #send goal
    sac.send_goal(goal)
    print("Sending goal:",x,y,w,z)
    #finish
    sac.wait_for_result()
    #print result
    print (sac.get_result())

def talker(coordinates):
    array = PoseArray()
    array.header.frame_id = 'map'
    array.header.stamp = rospy.Time.now()
    pose = Pose()
    pose.position.x = float(coordinates[0])
    pose.position.y = float(coordinates[1])
    pose.orientation.w = float(coordinates[2])
    pose.orientation.z = float(coordinates[3])
    array.poses.append(pose)

    pub = rospy.Publisher('simpleNavPoses', PoseArray, queue_size=100)
    rate = rospy.Rate(1) # 1hz

        #To not have to deal with threading, Im gonna publish just a couple times in the begging, and then continue with telling the robot to go to the points
    count = 0
    while count<1:
        rate.sleep()	
        print("sending rviz arrow")
        pub.publish(array)
        count +=1
def navigate2point(coordinates):
    try:
        simple_move((coordinates[0]),(coordinates[1]),(coordinates[2]),(coordinates[3]))
        talker(coordinates)
        print("goal reached")
    except rospy.ROSInterruptException:
        print ("Keyboard Interrupt")


if __name__ == "__main__":
    rospy.wait_for_service('get_reference_cloud_region')
    try:
        get_reference = rospy.ServiceProxy('get_reference_cloud_region', PointCloudWithEntropy)
        msg = get_reference(0)
        msg2pc(msg)
        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
