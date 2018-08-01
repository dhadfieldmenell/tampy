import rospy
from numpy_tutorial.msg import Train_data
from rospy.numpy_msg import numpy_msg
import cv2
from core.util_classes.pose_estimator import *
from core.util_classes.image_partition_helper import *

pose_predictor = create_net(net_type='partition_regression')
existence_predictor = create_net(net_type='partition_existence')
images_regression = []
labels_regression = []
images_existence = []
labels_existence = []
def callback(data):
    print "new msg"
    actual_image = data.image.reshape((640, 480)[::-1] + (3,))
    compressed_img = []
    for i in range(3): 
        temp_img = actual_image[:,:,i]
        new_img = cv2.resize(temp_img, dsize = (120, 160), interpolation=cv2.INTER_CUBIC)
        compressed_img.append(new_img)
    rgb_pic = np.array(compressed_img)
    # expecting shapes of (3, 160, 120)
    imgs_regr, labels_regr = img_partition_collector_regression(rgb_pic, data.label, iters=2000)
    imgs_exis, labels_exis = img_partition_collector_existence(rgb_pic, data.label)
    images_regression.extend(imgs_regr)
    labels_regression.extend(labels_regr)
    images_existence.extend(imgs_exis)
    labels_existence.extend(labels_exis)
    train(pose_predictor, np.array(images_regression).transpose((0, 2, 3, 1)), np.array(labels_regression), num_epochs=1)
    train(existence_predictor, np.array(images_existence), np.array(labels_existence).reshape((len(labels_existence),1)), num_epochs=1)
    print "finished training on new data"

def calculate_quadrant_pose(image_number):
    num_image_per_pic = 192 # 192 10x10 pics can be formed by a pic of 160x120
    x_lim = (-3, 10)
    y_lim = (-5, 10)
    top_left = (x_lim[0], y_lim[1])
    scaling_x = float(x_lim[1] - x_lim[0]) / 120
    scaling_y = float(y_lim[1] - y_lim[0]) / 160
    num_iter_x = image_number // 16
    num_iter_y = image_number % 16
    start_coord = (num_iter_x * 10, num_iter_y * 10)
    start_x = int(start_coord[0])
    start_y = int(start_coord[1])
    relative_coord = np.array([(start_x) * scaling_x + x_lim[0], - (start_y) * scaling_y + y_lim[1]]) # relative to top left
    return relative_coord

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("floats", numpy_msg(Train_data), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
