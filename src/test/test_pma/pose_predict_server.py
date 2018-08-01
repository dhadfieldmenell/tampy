#!/usr/bin/env python

from numpy_tutorial.srv import PosePredict, PosePredictResponse
from core.util_classes.pose_estimator import *
import rospy
import numpy as np

def handle_pose_predict_partition(req):
    print "new msg"
    pose_predictor = create_net(net_type='partition_regression')
    existence_predictor = create_net(net_type='partition_existence')
    actual_image = req.image.reshape((640, 480)[::-1] + (3,))
    compressed_img = []
    for i in range(3): 
        temp_img = actual_image[:,:,i]
        new_img = cv2.resize(temp_img, dsize = (120, 160), interpolation=cv2.INTER_CUBIC)
        compressed_img.append(new_img)
    rgb_pic = np.array(compressed_img)
    # expecting shapes of (3, 160, 120)
    imgs_exis, labels_exis = img_partition_collector_existence(rgb_pic, data.label)
    existence_preds = predict(existence_predictor, np.array(imgs_exis))
    image_number = 0
    for result in existence_preds:
        if result != 0.0:
            break
        image_number += 1
    relative_coord_quad = calculate_quadrant_pose(image_number)

    relative_pose_pred = predict(pose_predictor, imgs_exis[image_number])
    relative_pose = next(relative_pose_pred)['predicted_coordinate']
    pose_predicted = np.array([relative_pose[0] + relative_coord_quad[0], relative_pose[1] + relative_coord_quad[1]])
    return PosePredictResponse(pose_predicted)

def handle_pose_predict(req):
    print "new msg"
    actual_image = req.image.reshape((640, 480)[::-1] + (3,))
    compressed_img = []
    for i in range(3): 
        temp_img = actual_image[:,:,i]
        new_img = cv2.resize(temp_img, dsize = (120, 160), interpolation=cv2.INTER_CUBIC)
        compressed_img.append(new_img)
    rgb_pic = np.array(compressed_img)
    # expecting shapes of (3, 160, 120)
    pose_predictor = create_net(net_type='normal')
    relative_pose_pred = predict(pose_predictor, rgb_pic)
    relative_pose = next(relative_pose_pred)['predicted_coordinate']
    return PosePredictResponse(pose_predicted)

def pose_predict_server():
    rospy.init_node('pose_predict_server')
    s = rospy.Service('pose_predict', PosePredict, handle_pose_predict_partition)
    print "Ready to predict pose."
    rospy.spin()

if __name__ == "__main__":
    pose_predict_server()
