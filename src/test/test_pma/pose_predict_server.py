#!/usr/bin/env python

from numpy_tutorial.srv import PosePredict, PosePredictResponse
from core.util_classes.pose_estimator import *
import rospy
import numpy as np

def handle_pose_predict(req):
	a = np.array(req.image)
	print("Returning {}".format(a.shape))
	return PosePredictResponse(req.image)

def pose_predict_server():
    rospy.init_node('pose_predict_server')
    s = rospy.Service('pose_predict', PosePredict, handle_pose_predict)
    print "Ready to predict pose."
    rospy.spin()

if __name__ == "__main__":
    pose_predict_server()
