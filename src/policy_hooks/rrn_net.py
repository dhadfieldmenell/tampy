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


def get_loss_layer(mlp_out, action, precision, batch_size):
    """The loss layer used for the MLP network is obtained through this class."""
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision, batch_size=batch_size)


def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, precision


def get_lstm_input_layer(dim_input, dim_output, lstm_steps):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
    net_input = tf.placeholder("float", [None, lstm_steps, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, lstm_steps, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = [d.value for d in cur_top.get_shape().dims[1:]]
        in_shape.append(dimension_hidden[layer_step])
        cur_weight = init_weights(in_shape, name='w_' + str(layer_step))
        bias_shape = [dimension_hidden[layer_step]]
        if len(in_shape) > 1: bias_shape.insert(0, in_shape[0])
        cur_bias = init_bias(bias_shape, name='b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases


def get_lstm_cell(lstm_size, keep_prob, name):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size, name=name), output_keep_prob=keep_prob)


def get_stacked_lstm_cells(lstm_size, n_lstm_layers, keep_prob, name):
    return tf.contrib.run.MultiRNNCell([get_lstm_cell(lstm_size if type(lstm_size) is int else lstm_size[i], keep_prob, name+'_cell_{0}'.format(i)) for i in range(n_lstm_layers)], name)


def get_lstm_layers(lstm_size, n_lstm_layers, lstm_steps, inputs, keep_prob, batch_size):
    stacked_lstm = get_stacked_lstm_cells(lstm_size, n_lstm_layers, keep_prob, 'lstm_w')
    initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
    lstm_out, final_state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=inisital_state)

    return inital_state, final_state, lstm_out


def tf_lstm_network(dim_input, dim_output, batch_size, network_config=None):
    lstm_size = network_config['lstm_size'] if lstm_size in network_config else 40
    n_lstm_layers = network_config['n_lstm_layers'] if 'n_lstm_layers' in network_config else 1
    lstm_steps = network_config['lstm_steps'] if 'lstm_steps' in network_config else 5
    
    n_layers = 1 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else copy(network_config['dim_hidden'])
    dim_hidden.append(dim_output)

    nn_inputs, action, precision = get_lstm_input_layer(dim_input, dim_output, lstm_steps)

    init_state, final_state, lstm_out = get_lstm_layers(lstm_size, n_lstm_layers, lstm_steps, nn_inputs, batch_size)
    
    mlp_applied, weights, biases = get_mlp_layers(lstm_out, n_layers, dim_hidden)
    fc_vars = weights + biases

    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision, batch_size=batch_size)
 
    return TfMap.init_from_lists([nn_inputs, action, precision], [mlp_applied], [loss_out]), fc_vars, []
