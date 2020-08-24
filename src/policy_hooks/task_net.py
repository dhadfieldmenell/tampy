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


def multi_softmax_loss_layer(labels, logits, boundaries, precision=None):
    start = 0
    losses = []
    for i, (start, end) in enumerate(boundaries):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels[:,start:end], logits=logits[:, start:end], reduction=tf.losses.Reduction.NONE)
        if precision is not None:
            loss *= precision[:, i]
        # loss /= float(end - start)
        # losses.append(tf.reduce_mean(loss, axis=0))
        losses.append(loss)
    stacked_loss = tf.stack(losses, axis=0, name='softmax_loss_stack')
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


def get_loss_layer(mlp_out, task, boundaries, precision=None):
    """The loss layer used for the MLP network is obtained through this class."""
    return multi_softmax_loss_layer(labels=task, logits=mlp_out, boundaries=boundaries, precision=precision)


def tf_classification_network(dim_input=27, dim_output=2, batch_size=25, network_config=None, input_layer=None, eta=None):
    n_layers = 2 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    boundaries = network_config['output_boundaries']
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
    loss_out = get_loss_layer(mlp_out=scaled_mlp_applied, task=action, boundaries=boundaries, precision=precision)
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
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else copy(network_config['dim_hidden'])
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
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else copy(network_config['dim_hidden'])
    dim_hidden.append(2)

    nn_input, action, precision = get_input_layer(dim_input, 2)
    mlp_applied, weights_FC, biases_FC = get_mlp_layers(nn_input, n_layers, dim_hidden)
    prediction = tf.sigmoid(mlp_applied, 'sigmoid_activation')
    fc_vars = weights_FC + biases_FC
    loss_out = get_sigmoid_loss_layer(mlp_out=mlp_applied, labels=action)

    return TfMap.init_from_lists([nn_input, action, precision], [prediction], [loss_out]), fc_vars, []
