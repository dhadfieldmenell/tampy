from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.io import loadmat
from tensorflow.contrib.layers.python.layers import spatial_softmax
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
#TODO: learn how to load pretrained weights

# train_data = images[:8000]  # Returns np.array
# train_labels = labels[:8000]
# eval_data = images[8000:]  # Returns np.array
# eval_labels = labels[8000:]

def cnn_model_fn_normal(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # My matplot images are 120x160 pixels, and have three color channels
  input_layer = tf.reshape(features["x"], [-1, 120, 160, 3])

  # Convolutional Layer #1
  # Computes 64 features using a 7x7 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 3]
  # Output Tensor Shape: [batch_size, 120, 160, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 64]
  # Output Tensor Shape: [batch_size, 120, 160, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Spatial Softmax Layer #1
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  spatial = spatial_softmax(conv3)

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]

  # Fully connected layer #1
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  coord = tf.layers.dense(inputs=spatial, units=2)

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "predicted_coordinate": tf.identity(coord, name="pred_coord"), 
    # "actual": tf.identity(labels, name="labels"),
    # # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # # `logging_hook`.
    # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels=labels, predictions=coord)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss)
def cnn_model_fn_partition_existence(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # My matplot images are 10x10 pixels, and have three color channels
  input_layer = tf.reshape(features["x"], [-1, 10, 10, 3])

  # Convolutional Layer #1
  # Computes 64 features using a 7x7 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 3]
  # Output Tensor Shape: [batch_size, 120, 160, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 64]
  # Output Tensor Shape: [batch_size, 120, 160, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Spatial Softmax Layer #1
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  spatial = spatial_softmax(conv3)

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]

  # Fully connected layer #1
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  coord = tf.layers.dense(inputs=spatial, units=1)

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "predicted_coordinate": tf.identity(coord, name="pred_coord"), 
    # "actual": tf.identity(labels, name="labels"),
    "predicted_boolean": tf.round(coord, name="pred_bool")
    # # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # # `logging_hook`.
    # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels=labels, predictions=coord)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
     "accuracy": tf.metrics.accuracy(
       labels=labels, predictions=predictions['predicted_boolean'])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def cnn_model_fn_partition_regression(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # My matplot images are 10x10 pixels, and have three color channels
  input_layer = tf.reshape(features["x"], [-1, 10, 10, 3])

  # Convolutional Layer #1
  # Computes 64 features using a 7x7 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 3]
  # Output Tensor Shape: [batch_size, 120, 160, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 64]
  # Output Tensor Shape: [batch_size, 120, 160, 32]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Spatial Softmax Layer #1
  # Computes 16 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 160, 32]
  # Output Tensor Shape: [batch_size, 120, 160, 16]
  spatial = spatial_softmax(conv3)

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]

  # Fully connected layer #1
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  coord = tf.layers.dense(inputs=spatial, units=2)

  predictions = {
    # Generate predictions (for PREDICT and EVAL mode)
    "predicted_coordinate": tf.identity(coord, name="pred_coord"), 
    # "actual": tf.identity(labels, name="labels"),
    # # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # # `logging_hook`.
    # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels=labels, predictions=coord)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
#   eval_metric_ops = {
#      "accuracy": tf.metrics.accuracy(
#        labels=labels, predictions=predictions['predicted_boolean'])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss)

def create_net(net_type="regular"):
  # Create the Estimator
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
      session_config=tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=tf.GPUOptions(allow_growth=True,
                    per_process_gpu_memory_fraction=0.5)))
  if net_type == "regular":
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn_normal,
        config=run_config,
        model_dir="./pose_weights_regular")
  elif net_type == "partition_regression":
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn_partition_regression,
        config=run_config,
        model_dir="./pose_weights_partition_regression")   
  elif net_type == "partition_existence":
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn_partition_existence,
        config=run_config,
        model_dir="./pose_weights_partition_existence")
  else:
    return     
  return classifier

def train(classifier, train_data, train_labels, num_epochs=1):
  if len(train_data) == 1:
    print("Not enough data")
    return
  train_data = np.asarray(train_data, dtype=np.float32)
  train_labels = np.asarray(train_labels, dtype=np.float32)
  mean_r = np.mean(train_data, axis=0)
  std_r = np.std(train_data, axis=0)
  for color_index in range(3):
    mean = np.mean(train_data[:,:,:,color_index], axis=0)
    std = np.std(train_data[:,:,:,color_index], axis=0)
    std[std==0] = 1e-6
    train_data[:,:,:,color_index] -= mean
    train_data[:,:,:,color_index] /= std
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=10,
      num_epochs=num_epochs,
      shuffle=True)

  classifier.train(
      input_fn=train_input_fn)

def training_error(classifier, train_images, train_labels):
  # Evaluate the model and print results
  print("Training Error")
  evaluate_loss(train_images, train_labels)

def validation_error(classifier, valid_images, valid_labels):
  # Evaluate the model and print results
  print("Validation Error")
  evaluate_loss(valid_images, valid_labels)

def evaluate_loss(classifier, images, labels):
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": images},
      y=labels,
      num_epochs=1,
      shuffle=False,
      batch_size=10)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

def predict(classifier, images):
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":images},
      num_epochs=1,
      shuffle=False)
  predictions = classifier.predict(input_fn=predict_input_fn)
  return predictions
