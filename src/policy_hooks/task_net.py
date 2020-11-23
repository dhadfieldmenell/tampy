import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap
import numpy as np
from copy import copy

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))


def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))


def batched_matrix_vector_multiply(vector, matrix):
    """ computes x^T A in mini-batches. """
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision, batch_size):
    """ Math:  out = (action - mlp_out)'*precision*(action-mlp_out)
                    = (u-uhat)'*A*(u-uhat)"""
    scale_factor = tf.constant(2*batch_size, dtype='float')
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b))  # this last dot product is then summed, so we just the sum all at once.
    return uPu/scale_factor


def softmax_loss_layer(labels, logits):
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


def sigmoid_loss_layer(labels, logits, precision=None):
    if precision is None:
        return tf.losses.sigmoid_cross_entropy(labels, logits=logits)

    precision = tf.expand_dims(precision, [1])
    return tf.losses.sigmoid_cross_entropy(labels, logits=logits, weights=precision)


def td_loss_layer(rewards, logits, precision, targets, done, gamma=0.95):
    targets = tf.reduce_max(targets, axis=-2, name='target_max_reduce')
    t = rewards + gamma * (1. - done) * targets
    loss = tf.reduce_mean(tf.square(logits - tf.stop_gradient(t)), name='td_loss')
    return loss


def multi_softmax_loss_layer(labels, logits, boundaries, precision=None, scalar=True, scope=''):
    start = 0
    losses = []
    for i, (start, end) in enumerate(boundaries):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels[:,start:end], logits=logits[:, start:end], reduction=tf.losses.Reduction.NONE)
        if precision is not None:
            loss *= precision[:, i] / (1+tf.reduce_mean(precision[:,i]))
        # loss /= float(end - start)
        # losses.append(tf.reduce_mean(loss, axis=0))
        losses.append(loss)
    stacked_loss = tf.stack(losses, axis=0, name='softmax_loss_stack'+scope)
    if scalar:
        stacked_loss = tf.reduce_mean(stacked_loss, axis=1)
    return stacked_loss # tf.reduce_sum(stacked_loss, axis=0)


def multi_mix_loss_layer(labels, logits, boundaries):
    start = 0
    losses = []
    for start, end in boundaries:
        losses.append(tf.losses.softmax_cross_entropy(onehot_labels=labels[:,start:end], logits=logits[:, start:end]))
    losses.append(tf.nn.l2(onehot_labels=labels[:,end:], logits=logits[:, end:]))
    stacked_loss = tf.stack(losses, axis=0, name='softmax_loss_stack')
    return tf.reduce_sum(stacked_loss, axis=0)


def sotfmax_prediction_layer(logits):
    return tf.nn.softmax(logits, name="softmax_layer")


def multi_sotfmax_prediction_layer(logits, boundaries):
    start = 0
    predictions = []
    for i in range(len(boundaries)):
        start, end = boundaries[i]
        predictions.append(tf.nn.softmax(logits[:,start:end], name="softmax_layer_{0}".format(i)))
    return tf.concat(predictions, axis=1, name='multi_softmax_layer')


def multi_mix_prediction_layer(logits, boundaries):
    start = 0
    predictions = []
    for i in range(len(boundaries)):
        start, end = boundaries[i]
        predictions.append(tf.nn.softmax(logits[:,start:end], name="softmax_layer_{0}".format(i)))
    predictions.append(logits[:,end:], name="euc_layer")
    return tf.concat(predictions, axis=1, name='multi_softmax_layer')


def get_input_layer(dim_input, dim_output, ndims=1):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    if type(dim_input) is int:
        net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    else:
        net_input = tf.placeholder("float", [None]+dim_input, name='nn_input')
    task = tf.placeholder('float', [None, dim_output], name='task')
    precision = tf.placeholder('float', [None, ndims], name='precision')
    return net_input, task, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden, offset=0, nonlin=False):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[-1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step+offset))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step+offset))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1 or nonlin:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.tensordot(cur_top, cur_weight, 1) + cur_bias)
        else:
            cur_top = tf.tensordot(cur_top, cur_weight, 1) + cur_bias

    return cur_top, weights, biases


def get_sigmoid_loss_layer(mlp_out, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=mlp_out, name='sigmoid_loss')


def get_loss_layer(mlp_out, task, boundaries, precision=None, scalar=True, scope=''):
    """The loss layer used for the MLP network is obtained through this class."""
    return multi_softmax_loss_layer(labels=task, logits=mlp_out, boundaries=boundaries, precision=precision, scalar=scalar, scope=scope)


def tf_classification_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None, eta=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    boundaries = network_config.get('output_boundaries', [(0, dim_output)])
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    dim_hidden.append(dim_output)

    if input_layer is None:
        nn_input, action, precision = get_input_layer(dim_input, dim_output, len(boundaries))
    else:
        nn_input, action, precision = input_layer 
    fc_input = nn_input
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    scaled_mlp_applied = mlp_applied
    if eta is not None:
        scaled_mlp_applied = mlp_applied * eta
    prediction = multi_sotfmax_prediction_layer(scaled_mlp_applied, boundaries)
    fc_vars = weights_FC + biases_FC
    loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision)
    return TfMap.init_from_lists([fc_input, action, precision], [prediction], [loss_out]), fc_vars, []


def tf_balanced_classification_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None, eta=None, class_tensor=None, class1_wt=0.5, class2_wt=0.5):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    boundaries = network_config.get('output_boundaries', [(0, dim_output)])
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output, len(boundaries))
    fc_input = nn_input
    if input_layer is not None:
        nn_input = tf.concat(input_layer, nn_input)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    scaled_mlp_applied = mlp_applied
    if eta is not None:
        scaled_mlp_applied = mlp_applied * eta
    prediction = multi_sotfmax_prediction_layer(scaled_mlp_applied, boundaries)
    fc_vars = weights_FC + biases_FC
    if class_tensor is None:
        loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision)
    else:
        loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision, scalar=False, scope='_1')
        loss_out_2 = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision, scalar=False, scope='_2')
        loss_out = class1_wt * class_tensor * tf.transpose(loss_out) + class2_wt * (1 - class_tensor) * tf.transpose(loss_out_2)
        loss_out = tf.reduce_mean(loss_out, axis=0)
    return TfMap.init_from_lists([fc_input, action, precision], [prediction], [loss_out]), fc_vars, []


def tf_mixed_classification_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else copy(network_config['dim_hidden'])
    dim_hidden.append(dim_output)
    boundaries = network_config['output_boundaries']

    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    prediction = multi_mix_prediction_layer(mlp_applied, boundaries)
    fc_vars = weights_FC + biases_FC
    loss_out = multi_mix_loss_layer(mlp_out=mlp_applied, task=action, boundaries=boundaries)

    return TfMap.init_from_lists([nn_input, action, precision], [prediction], [loss_out]), fc_vars, []


def tf_cond_classification_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None, eta=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    boundaries = network_config['output_boundaries']
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    # dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output, len(boundaries))
    offset = len(dim_hidden)
    fc_input = nn_input
    if input_layer is not None:
        nn_input = tf.concat(input_layer, nn_input)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers-1, dim_hidden, nonlin=True)
    pred = []
    cur_input = mlp_applied
    for i, (st, en) in enumerate(boundaries):
        dh = [dim_hidden[-1], en-st]
        mlp_applied, weights_FC, biases_FC = get_mlp_layers(cur_input, len(dh), dh, offset)
        pred.append(mlp_applied)
        cur_input = tf.concat([cur_input, tf.stop_gradient(mlp_applied)], axis=1)
        offset += len(dh)

    mlp_applied = tf.concat(pred, axis=1)
    scaled_mlp_applied = mlp_applied
    # prediction = multi_sotfmax_prediction_layer(mlp_applied, boundaries)
    # loss_out = get_loss_layer(mlp_out=mlp_applied, task=action, boundaries=boundaries)

    if eta is not None:
        scaled_mlp_applied = mlp_applied * eta
    prediction = multi_sotfmax_prediction_layer(scaled_mlp_applied, boundaries)
    fc_vars = weights_FC + biases_FC
    loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision)

    return TfMap.init_from_lists([fc_input, action, precision], [prediction], [loss_out]), fc_vars, []


def tf_value_network(dim_input=27, dim_output=1, batch_size=25, network_config=None, input_layer=None, target=None, done=None, imwt=0.):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    dim_hidden.append(dim_output)


    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    fc_input = nn_input
    if input_layer is not None:
        nn_input = tf.concat([input_layer, nn_input], axis=1)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    prediction = mlp_applied # tf.nn.sigmoid(mlp_applied, 'sigmoid_activation')
    fc_vars = weights_FC + biases_FC
    if target is None:
        loss_out = prediction # sigmoid_loss_layer(action, prediction, precision=precision)
    else:
        if imwt < 1e-3:
            loss_out = td_loss_layer(action, prediction[:,0], precision=precision, targets=target, done=done)
        elif imwt > 1-1e-3:
            loss_out = tf.reduce_mean(tf.square(action - prediction[:,0]))
        else:
            loss1 = td_loss_layer(action, prediction[:,0], precision=precision, targets=target, done=done)
            loss2 = tf.reduce_mean(tf.square(action - prediction[:,0]))
            loss_out = (1 - imwt) * loss1 + imwt * loss2

    return TfMap.init_from_lists([fc_input, action, precision], [prediction], [loss_out]), fc_vars, []


def tf_binary_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    dim_hidden.append(dim_output)


    nn_input, action, precision = get_input_layer(dim_input, 2)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    prediction = tf.sigmoid(mlp_applied, 'sigmoid_activation')
    fc_vars = weights_FC + biases_FC
    loss_out = get_sigmoid_loss_layer(mlp_out=mlp_applied, labels=action)

    return TfMap.init_from_lists([nn_input, action, precision], [prediction], [loss_out]), fc_vars, []


def multi_modal_class_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None, eta=None):
    """
    An example a network in tf that has both state and image inputs, with the feature
    point architecture (spatial softmax + expectation).
    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        A tfMap object that stores inputs, outputs, and scalar loss.
    """
    pool_size = 2
    filter_size = 5
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = network_config.get('dim_hidden', 40)
    if type(dim_hidden) is int:
        dim_hidden = (n_layers - 1) * [dim_hidden]
    else:
        dim_hidden = copy(dim_hidden)
    dim_hidden.append(dim_output)
    boundaries = network_config['output_boundaries']


    # List of indices for state (vector) data and image (tensor) data in observation.
    x_idx, img_idx, i = [], [], 0
    for sensor in network_config['obs_include']:
        dim = network_config['sensor_dims'][sensor]
        if sensor in network_config['obs_image_data']:
            img_idx = img_idx + list(range(i, i+dim))
        else:
            x_idx = x_idx + list(range(i, i+dim))
        i += dim

    if input_layer is None:
        nn_input, action, precision = get_input_layer(dim_input, dim_output, len(boundaries))
    else:
        nn_input, action, precision = input_layer

    state_input = nn_input[:, 0:x_idx[-1]+1]
    image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]

    # image goes through 3 convnet layers
    num_filters = network_config['num_filters']
    n_conv = len(num_filters)

    im_height = network_config['image_height']
    im_width = network_config['image_width']
    num_channels = network_config['image_channels']
    image_input = tf.reshape(image_input, [-1, num_channels, im_width, im_height])
    image_input = tf.transpose(image_input, perm=[0,3,2,1])

    # we pool twice, each time reducing the image size by a factor of 2.
    #conv_out_size = int(im_width/(2.0*pool_size)*im_height/(2.0*pool_size)*num_filters[1])
    #first_dense_size = conv_out_size + len(x_idx)

    # Store layers weight & bias
    weights = {}
    biases = {}
    conv_layers = []
    cur_in = num_channels
    cur_in_layer = image_input
    for i in range(n_conv):
        weights['conv_wc{0}'.format(i)] = init_weights([filter_size, filter_size, cur_in, num_filters[i]], name='conv_wc{0}'.format(i)) # 5x5 conv, 1 input, 32 outputs
        biases['conv_bc{0}'.format(i)] = init_bias([num_filters[i]], name='conv_bc{0}'.format(i))
        cur_in = num_filters[i]
        if i == 0:
            conv_layers.append(conv2d(img=cur_in_layer, w=weights['conv_wc{0}'.format(i)], b=biases['conv_bc{0}'.format(i)], strides=[1,2,2,1]))
        else:
            conv_layers.append(conv2d(img=cur_in_layer, w=weights['conv_wc{0}'.format(i)], b=biases['conv_bc{0}'.format(i)]))
        cur_in_layer = conv_layers[-1]

    _, num_rows, num_cols, num_fp = conv_layers[-1].get_shape()
    num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
    x_map = np.empty([num_rows, num_cols], np.float32)
    y_map = np.empty([num_rows, num_cols], np.float32)

    for i in range(num_rows):
        for j in range(num_cols):
            x_map[i, j] = (i - num_rows / 2.0) / num_rows
            y_map[i, j] = (j - num_cols / 2.0) / num_cols

    x_map = tf.convert_to_tensor(x_map)
    y_map = tf.convert_to_tensor(y_map)

    x_map = tf.reshape(x_map, [num_rows * num_cols])
    y_map = tf.reshape(y_map, [num_rows * num_cols])

    # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
    features = tf.reshape(tf.transpose(conv_layers[-1], [0,3,1,2]),
                          [-1, num_rows*num_cols])
    softmax = tf.nn.softmax(features)

    fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
    fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

    fp = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])

    fc_input = tf.concat(axis=1, values=[fp, state_input])

    fc_output, weights_FC, biases_FC = get_mlp_layers(fc_input, n_layers, dim_hidden)
    fc_vars = weights_FC + biases_FC
    last_conv_vars = fc_input

    scaled_mlp_applied = fc_output
    if eta is not None:
        scaled_mlp_applied = fc_output * eta
    prediction = multi_sotfmax_prediction_layer(scaled_mlp_applied, boundaries)
    loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision)
    return TfMap.init_from_lists([nn_input, action, precision], [prediction], [loss_out]), fc_vars, last_conv_vars


def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_xavier_weights(filter_shape, poolsize=(2, 2)):
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))



