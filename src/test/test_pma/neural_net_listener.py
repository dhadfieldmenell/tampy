import rospy
from numpy_tutorial.msg import Train_data
from rospy.numpy_msg import numpy_msg
from core.util_classes.pose_estimator import *

pose_predictor = create_net()
def callback(data):
	print "new msg"
	print rospy.get_name(), "I heard %s"%str(data.image)
	print rospy.get_name(), "I heard %s"%str(data.label)
	actual_image = data.image.reshape((640, 480)[::-1] + (3,))
	print actual_image.shape

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("floats", numpy_msg(Train_data), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()