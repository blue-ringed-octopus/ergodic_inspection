#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:17:49 2023

@author: barc
"""
from numpy import cos,sin, arctan2
import numpy as np
def angle_wrapping(theta):
    return arctan2(sin(theta), cos(theta))


def v2t(x):
    return np.array([[cos(x[2]), -sin(x[2]),0, x[0]],
                     [sin(x[2]), cos(x[2]), 0, x[1]],
                     [0, 0, 1, 0],
                      [0, 0, 0,1]])
def t2v(T):
    return np.array([T[0,3], T[1,3], arctan2(T[1,0], T[0,0])])