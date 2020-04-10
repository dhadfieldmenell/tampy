USE_OPENRAVE = True
USE_ROS = True

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

