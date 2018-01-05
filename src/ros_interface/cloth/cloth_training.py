
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, GaussianDropout, GaussianNoise, GlobalAveragePooling2D, Input
from keras.optimizers import Adam, Nadam, RMSprop
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping

import tensorflow as tf

import cv2

import numpy as np

import os

path = './'
images_file = 'data/shuffledClothGridImages.npy'
labels_file = 'data/shuffledClothGridLabels.npy'

im_height = 10
im_width = 10

input_images = np.load(path+images_file)
labels = np.load(path+labels_file)

# images = preprocess_input(np.load(path+images_file)[:,:,:,:3], mode='tf')
# test_images = preprocess_input(np.load(path+test_image_file)[:,:,:,:3], mode='tf')
# images = np.load(path+images_file)[:,:,:,:3]
# test_images = np.load(path+test_image_file)[:,:,:,:3]

#test_images = (test_images - np.mean(images)) / np.std(images)
#images = (images - np.mean(images)) / np.std(images)

# input_images = np.ndarray((len(images), im_height, im_width, 3))
# for i in range(len(images)):
#    input_images[i]  = cv2.resize(images[i], (im_width, im_height))

# test_input_images = np.ndarray((len(test_images), im_height, im_width, 3))
# for i in range(len(test_images)):
#     test_input_images[i] = cv2.resize(test_images[i], (im_width, im_height))

# labels = np.load(path+labels_file)
# labels = np.c_[labels[:,:2]-labels[:,4:6], labels[:,7]]

# test_labels = np.load(path+test_label_file)
# test_labels = np.c_[test_labels[:,:2]-test_labels[:,4:6], test_labels[:,7]]
# labels[:,1] -= 0.325
# x_vals = labels[:,0].copy()
# y_vals = labels[:,1].copy()
# angles = -1.0 * np.arccos(y_vals / np.sqrt(x_vals**2+y_vals**2)) + labels[:,2] + np.pi/2.0
# labels[:,0] = np.cos(angles) * np.sqrt(x_vals**2 + y_vals**2)
# labels[:,1] = np.sin(angles) * np.sqrt(x_vals**2 + y_vals**2)

# test_labels[:,1] -= 0.325

training_set = input_images[50:]
training_labels = labels[50:]

mean = np.mean(training_set, axis=(0,1,2))
std = np.std(training_set, axis=(0,1,2))

training_set = (training_set - mean) / std

validation_set = (input_images[:50] - mean) / std
validation_labels = labels[:50]

# def input_generator(images, labels, batch_size, steps):
#     i = 0
#     while i < steps*batch_size:
#         next_ind = i % len(images)
#         yield (images[next_ind:next_ind+batch_size], labels[next_ind:next_ind+batch_size])
#         i += batch_size
#         if i % len(images) < batch_size:
#             new_order = np.random.permutation(len(images))
#             images = images[new_order]
#             labels = labels[new_order]

# preprocessing_fn = lambda x: x - [103.939, 116.779, 123.68]  # lambda x: 2*((x / 255.0) - 0.5)
augment_gen = image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=180, shear_range=np.pi/16, zoom_range=0.0, channel_shift_range=0.1)
# augment_gen.fit(training_set[:10])

def get_session(gpu_fraction=0.5, allow_growth=True):
    num_threads = 0 # os.environ.get('OMP_NUM_THREADS')
    if allow_growth:
        gpu_options = tf.GPUOptions(allow_growth=True)
    else:
        gpu_options = tf.GPUOptions(gpu_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))

K.tensorflow_backend.set_session(get_session())

inputs = Input(shape=(im_height, im_width, 3))
# x = GaussianNoise(0.0075)(x)
# x = GaussianNoise(0.01)(x)

# create the base pre-trained model
# base_model = InceptionV3(input_shape=(im_height, im_width, 3), weights='imagenet', include_top=False)
# base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
# base_model = InceptionV3(input_tensor=x, weights=None, include_top=False)
# x = base_model.output
# x = GlobalAveragePooling2D()(inputs)
x = Flatten()(inputs)
# x = GaussianNoise(0.001)(x)
# let's add a fully-connected layer
# x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dense(32, activation='relu')(x)
# x = Dense(1000, activation='relu')(x)
# x = Dense(500, activation='relu')(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(1, activation=None)(x)

# this is the model we will train
model = Model(inputs=inputs, outputs=predictions)
# model = load_model('nov18wristtrained2.h5')
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer=Adam(lr=1e-5, decay=0.995), loss='mean_absolute_error')
# model.compile(optimizer=RMSprop(5e-8), loss='mean_squared_error')
model.compile(optimizer=Adam(1e-2), loss='mean_absolute_error')
# model.compile(optimizer=Nadam(1e-5), loss='mean_absolute_error')
batch_size = 50
epoch = 1000


# train the model on the new data for a few epochs
# model.fit_generator(input_generator(training_set, training_labels, batch_size, 10000000), steps_per_epoch=epoch/batch_size, epochs=4000, validation_data=input_generator(validation_set, validation_labels, batch_size, 10000000), validation_steps=len(validation_set)/batch_size)

for i in range(5):
    model.fit_generator(augment_gen.flow(training_set, training_labels, batch_size), 
                        steps_per_epoch=epoch/batch_size,
                        epochs=250,
                        validation_data=augment_gen.flow(validation_set, validation_labels, batch_size),
                        validation_steps=len(validation_set)/batch_size,
                        max_queue_size=50)
    import ipdb; ipdb.set_trace()

