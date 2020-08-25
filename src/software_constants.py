USE_OPENRAVE = False
USE_ROS = False

try:
    import rospy
except Exception as e:
    print(e)
    USE_ROS = False

try:
    import openravepy
except Exception as e:
    print(e)
    USE_OPENRAVE = False
