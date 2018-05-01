import baxter_interface
import rospy
from sensor_msgs.msg import Int8

from ros_interface.laundry_env_monitor import *

rospy.init_node('laundry_shutdown_node')

pub = rospy.Publisher('/execution_state', Int8, queue_size=1)
