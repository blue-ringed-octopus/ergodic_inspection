# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:53:01 2024

@author: hibad
"""


from __future__ import print_function

from ergodic_inspection.srv import PointCloudWithEntropy, PointCloudWithEntropyResponse
import rospy
def handle_add_two_ints(req):
     print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
     return PointCloudWithEntropyResponse(req.a + req.b)
 
def add_two_ints_server():
     rospy.init_node('add_two_ints_server')
     s = rospy.Service('add_two_ints', PointCloudWithEntropy, handle_add_two_ints)
     print("Ready to add two ints.")
     rospy.spin()
 
if __name__ == "__main__":
     add_two_ints_server()