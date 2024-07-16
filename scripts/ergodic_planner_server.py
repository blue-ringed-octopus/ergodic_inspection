# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:49:39 2024

@author: hibad
"""

from ergodic_planner import Ergodic_Planner
import rospy
from ergodic_inspection.srv import GetGraphStructure


if __name__ == '__main__':
    rospy.wait_for_service('GetGraphStructure')
    get_graph = rospy.ServiceProxy('GetGraphStructure', GetGraphStructure)
    msg = get_graph(1)
    print(msg)
    # region_planner = Ergodic_Planner(map_manager.hierarchical_graph.levels[1])
