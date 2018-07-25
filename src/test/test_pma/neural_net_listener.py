import rospy
from numpy_tutorial.msg import Train_data
from rospy.numpy_msg import numpy_msg
from core.util_classes.pose_estimator import *
import cv2

pose_predictor = create_net()
images = []
labels = []
def callback(data):
	print "new msg"
	actual_image = data.image.reshape((640, 480)[::-1] + (3,))
        compressed_img = []
        for i in range(3): 
            temp_img = actual_image[:,:,i]
            new_img = cv2.resize(temp_img, dsize = (160, 120), interpolation=cv2.INTER_CUBIC)
            compressed_img.append(new_img)
        rgb_pic = np.array(compressed_img)
	images.append(rgb_pic)
	labels.append(data.label)
	train(pose_predictor, np.array(images), np.array(labels), num_epochs=100)
	print "finished training on new data"

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("floats", numpy_msg(Train_data), callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
