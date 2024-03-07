# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:25:49 2024

@author: hibad
"""
import numpy as np
from numpy import arccos, sin, cos, trace
Module SO3:
    @staticmethod
    def vee_SO3(W):
        return np.array([W[2,1], W[0,2], W[1,0]])
    
    @staticmethod
    def hat_SO3(w):
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])
    
    @staticmethod
    def Log_SO3(R):
        theta=arccos((trace(R)-1)/2)
        if theta == 0:
            return np.zeros(3)
        u=theta*vee_SO3((R-R.T))/(2*sin(theta))
        return u
    
    @staticmethod
    def Exp_SO3(u):
        theta=np.linalg.norm(u)
        if theta==0: 
            return np.eye(3)
        u=u/theta
        R=np.eye(3)+sin(theta)*hat_SO3(u)+(1-cos(theta))*np.linalg.matrix_power(hat_SO3(u),2)
        return R

    def Jl_inv_SO3(w):
        t=norm(w)
        J=np.eye(3)-1/2*hat_SO3(w)+(1/t**2-(1+cos(t))/(2*t*sin(t)))*hat_SO3(w)@hat_SO3(w)
        return J

    def Jr_inv_SO3(w):
        t=norm(w)
        J=np.eye(3) + 1/2*hat_SO3(w) + (1/t**2-(1+cos(t))/(2*t*sin(t))) * (hat_SO3(w)@hat_SO3(w))
        return J

    def Jr_SO3(theta):
        t= norm(theta)
        if t==0:
            return np.eye(3)
        c = cos(t)
        s = sin(t)
        theta_cross = hat_SO3(theta)
        return np.eye(3) - ((1-c)/t**2)*theta_cross + (t-s)/t**3* theta_cross@theta_cross 