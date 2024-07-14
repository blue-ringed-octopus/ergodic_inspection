# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:49:39 2024

@author: hibad
"""

from ergodic_planner import Ergodic_Planner
import rospy

if __name__ == '__main__':
    rospy.wait_for_service('set_entropy')
    region_planner = Ergodic_Planner(map_manager.hierarchical_graph.levels[1])
