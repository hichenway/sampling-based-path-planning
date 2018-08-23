"""
Parts of the code in this file are from https://github.com/yrouben/Sampling-Based-Path-Planning-Library. Thanks for the coder yrouben.
"""

from __future__ import division
from shapely.geometry import Polygon
from environment import Environment
from RRTs import RRTPlanner
from matplotlib import pyplot as plt


environment = Environment('bugtrap.yaml')
bounds = (-2, -3, 12, 8)
start_pose = (2, 2.5)
goal_region = Polygon([(10,5), (10,6), (11,6), (11,5)])
object_radius = 0.3
steer_distance = 0.3
num_iterations = 10000
resolution = 3
drawResults = True
runForFullIterations = False


sbpp = RRTPlanner()
path= sbpp.RRT(environment, bounds, start_pose, goal_region, object_radius, steer_distance, num_iterations, resolution, drawResults, runForFullIterations)
plt.show()

