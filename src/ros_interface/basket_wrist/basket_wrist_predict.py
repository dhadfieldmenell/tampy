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
        self.model = load_model(TRAINED_MODEL)
        # self.model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=['mae'])
        # self.model._make_predict_function()
        # f = open("ros_interface/basket_wrist/nov11arch.json", "r")
        # self.model = model_from_json(f.read())
        # f.close()
        # self.model.load_weights("ros_interface/basket_wrist/nov11weights.h5")

    def predict(self, image):
        processed_input = preprocess_input(image, mode='tf')
        processed_input = cv2.resize(processed_input, (144, 144))
        processed_input = processed_input.reshape((1, 144, 144, 3))
        return self.model.predict(processed_input).flatten()
