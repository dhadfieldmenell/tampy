import rospy

from ros_interface.laundry_env_monitor import LaundryEnvironmentMonitor
# from keras import backend as K

# K.set_learning_phase(0)

rospy.init_node("Laundry_Node")
lem = LaundryEnvironmentMonitor()
# print lem.move_basket_to_washer()
# print lem.move_basket_from_washer()
# # import ipdb; ipdb.set_trace()
# lem.load_washer_from_basket()
# lem.load_washer_from_region_1()
# lem.load_basket_from_region_2()
lem.open_washer()
# lem.close_washer()
# lem.reset_laundry()
# lem.run_baxter()
# lem.unload_washer_into_basket()
# lem.fold_cloth()
import ipdb; ipdb.set_trace()
