# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:20:20 2024

@author: hibad
"""

import pyrealsense2 as rs
import numpy as np
import cv2
def get_rs_param(cfg):
    profile = cfg.get_stream(rs.stream.depth)
    intr = profile.as_video_stream_profile().get_intrinsics()
    return [intr.fx, intr.fy, intr.ppx, intr.ppy]

TAG_SIZE=0.06

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align = rs.align(rs.stream.color)

cfg=pipeline.start(config)
cam_param=get_rs_param(cfg)

frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)

aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
color_frame = aligned_frames.get_color_frame()

depth_image = np.asanyarray(aligned_depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
images = np.hstack((bg_removed, depth_colormap))

cv2.imshow("rgb", depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()