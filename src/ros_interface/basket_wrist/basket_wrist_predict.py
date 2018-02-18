from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, model_from_json
from keras import backend as K

import tensorflow as tf

import cv2

import numpy as np

import json

TRAINED_MODEL = 'ros_interface/basket_wrist/jan25TrainedBasketWrist.h5'

def get_session(use_gpu=False, gpu_fraction=0.5, allow_growth=True):
    num_threads = 0 # os.environ.get('OMP_NUM_THREADS')
    if not use_gpu:
        config = tf.ConfigProto(device_count={'GPU': 0})
        return tf.Session(config=config)
    
    if allow_growth:
        gpu_options = tf.GPUOptions(allow_growth=True)
    else:
        gpu_options = tf.GPUOptions(gpu_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))

class BasketWristPredict:
    def __init__(self):
        # K.tensorflow_backend.set_session(get_session())
        # self.graph = tf.get_default_graph()
        with tf.device("/cpu:0"):
            self.model = load_model(TRAINED_MODEL)
        # self.model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=['mae'])
        # self.model._make_predict_function()
        # f = open("ros_interface/basket_wrist/nov11arch.json", "r")
        # self.model = model_from_json(f.read())
        # f.close()
        # self.model.load_weights("ros_interface/basket_wrist/nov11weights.h5")
        self.cur_im = None
        self.image_sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image, self.callback)

    def callback(self, data):
        self.cur_im = data

    def predict(self, ee_pos):
        im = self.bridge.imgmsg_to_cv2(self.cur_im, 'passthrough')
        im = np.array(im, dtype=np.float32)
        processed_input = preprocess_input(im, mode='tf')
        processed_input = cv2.resize(processed_input, (144, 144))
        processed_input = processed_input.reshape((1, 144, 144, 3))
        preds = self.model.predict(processed_input).flatten()
        x_pos = ee_pos[0] + preds[0]
        y_pos = ee_pos[1] + preds[1]
        theta = preds[2]
        return np.array([x_pos, y_pos, theta])
        