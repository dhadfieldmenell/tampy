import numpy as np
import tensorflow as tf


def conv2d(inputs, n_filters, filter_size, padding, name, reuse, strides=1, var_scale=1.):
    with tf.variable_scope(name, reuse=reuse):
        weights = var_scale * tf.get_variable(name='conv_w', shape=[filter_size[0], filter_size[1], inputs.shape[-1].value, n_filters])
        bias = tf.get_variable(name='conv_b', initializer=tf.zeros([n_filters]))
        conv = tf.nn.conv2d(inputs, padding=padding.upper(), filter=weights, strides=[1,strides,strides,1])

    return tf.nn.bias_add(conv, bias)

# def conv2d_transpose(inputs, n_filters, filter_size, padding, name, reuse, strides=[1,1,1,1], var_scale=1.):
#     with tf.variable_scope(name, reuse=reuse):
#         weights = var_scale * tf.get_variable(name='conv_w', shape=[filter_size[0], filter_size[1], n_filters, inputs.shape[-1].value])
#         bias = tf.get_variable(name='conv_b', initializer=tf.zeros([n_filters]))
#         conv = tf.nn.conv2d_transpose(inputs, padding=padding.upper(), filter=weights, strides=strides, output_shape=[])

#     return tf.nn.bias_add(conv, bias)

def dense(inputs, units, name, reuse, bias_initializer=None, var_scale=1.):
    with tf.variable_scope(name, reuse=reuse):
        weights = var_scale * tf.get_variable(name='dense_w', shape=[inputs.shape[-1].value, units])
        bias = tf.get_variable(name='dense_b', initializer=tf.zeros([units]))
    return tf.nn.bias_add(tf.matmul(inputs, weights), bias)


class Encoder(object):
    def get_net(self, x_in, training, n_channels=None, filter_sizes=None, strides=None, fc_dims=None, fc_in=None, reuse=False, config=None):
        import tensorflow as tf
        if config is not None:
            n_channels = config['n_channels']
            filter_sizes = config['filter_sizes']
            strides = config['strides']
            fc_dims = config.get('fc_dims', fc_dims)

        out = x_in
        last_conv_shape = None
        with tf.variable_scope('encoder', reuse=reuse):
            if n_channels is not None:
                n_conv_layers = list(range(len(n_channels)))
                for n_c, fs, s, i in zip(n_channels, filter_sizes, strides, n_conv_layers):
                    out = conv2d(out, n_c, (fs, fs), padding='same', strides=s, name='encode_conv{0}'.format(i), reuse=reuse)
                    # out = tf.layers.batch_normal/zization(out, training=training, name='batch_encoder{0}'.format(i))
                    out = tf.nn.relu(out)

            last_conv_shape = out.shape

            if fc_dims is not None:
                out = tf.reshape(out, [-1, np.prod([out.shape[i].value for i in range(1, 4)])])
                if fc_in is not None:
                    out = tf.concat([out, fc_in], axis=-1)

                n_dense_layers = list(range(len(fc_dims)))
                for d, i in zip(fc_dims, n_dense_layers):
                    if i == len(fc_dims) - 1:
                        d = 2 * d # Need to get mu and std
                    out = dense(out, d, 'encode_dense{0}'.format(i), reuse)
                    if i < len(fc_dims) - 1:
                        # out = tf.layers.batch_normalization(out, training=training, name='dense_batch_encoder{0}'.format(i))
                        out = tf.nn.relu(out)
            else:
                out = conv2d(out, n_channels[-1], (3, 3), padding='same', strides=1, name='encode_conv_out', reuse=reuse)

            mu, logvar = tf.split(out, 2, axis=-1)
            if config.get('out_act', None) == 'tanh':
                logvar = tf.nn.tanh(logvar)
        return mu, logvar + 1e-6


class Decoder(object):
    def get_net(self, x_in, training, conv_init_shape=None, n_channels=None, filter_sizes=None, strides=None, fc_dims=None, fc_out_size=None, bernoulli=True, reuse=False, config=None):
        import tensorflow as tf
        if config is not None:
            n_channels = config['n_channels']
            filter_sizes = config['filter_sizes']
            strides = config['strides']
            fc_dims = config.get('fc_dims', fc_dims)
            fc_out_size = config.get('fc_out_size', fc_out_size)
            conv_init_shape = config['conv_init_shape']

        out = x_in
        with tf.variable_scope('decoder', reuse=reuse):
            if fc_dims is not None:
                fc_dims[-1] = conv_init_shape[0]*conv_init_shape[1]*conv_init_shape[2]
                for i, d in enumerate(fc_dims):
                    out = dense(out, d, 'decode_dense{0}'.format(i), reuse)
                    # out = tf.layers.batch_normalization(out, training=training, name='dense_batch_decoder{0}'.format(i))
                    if i < len(fc_dims) - 1:
                        out = tf.nn.relu(out)

            if fc_out_size is not None:
                fc_out = out[-fc_out_size:]
                out = out[:-fc_out_size]
                out = tf.nn.relu(out)

            out = dense(out, conv_init_shape[0]*conv_init_shape[1]*conv_init_shape[2], 'decode_to_conv_init', reuse)
            out = tf.nn.relu(out)
            out = tf.reshape(out, (-1, conv_init_shape[0], conv_init_shape[1], conv_init_shape[2]))

            if n_channels is not None:
                n_deconv_layers = list(range(len(n_channels)))
                for n_c, fs, s, i in zip(n_channels, filter_sizes, strides, n_deconv_layers):
                    if i == len(n_channels)-1:
                        n_c *= 2
                    out = tf.layers.conv2d_transpose(out, n_c, (fs, fs), padding='valid', strides=s, name='decode_deconv{0}'.format(i), reuse=reuse)
                    # out = tf.layers.batch_normalization(out, training=training, name='batch_decoder{0}'.format(i))
                    if i < len(n_channels) - 1:
                        out = tf.nn.relu(out)

            mu, logvar = tf.split(out, 2, axis=-1)
            if config.get('out_act', None) == 'sigmoid':
                # mu = tf.nn.sigmoid(mu)
                logvar = tf.nn.tanh(logvar)
        return mu, logvar + 1e-6


class LatentDynamics(object):
    def get_net(self, x_in, task_in, training, fc_dims=None, reuse=False, config=None):
        import tensorflow as tf
        if config is not None:
            fc_dims = config.get('fc_dims', fc_dims)

        self.weights = []
        with tf.variable_scope('latent_dynamics', reuse=reuse):
            out = tf.concat([x_in, task_in], axis=-1)
            if len(out.shape) > 2:
                # for i, n, fs in zip(range(len(channels)), n_channels, filter_sizes):
                #     out = conv2d(out, n, fs, padding='SAME', name='conv_{0}'.fomrat(i), reuse=reuse)
                #     out = tf.nn.relu(out)
                # out = conv2d(out, 2*x_in.shape[-1], 2, padding='SAME', name='dynamics_out', reuse=reuse)
                # out_mu, out_logvar = tf.split(out, 2, -1)
                out = tf.reshape(out, [-1, np.prod([out.shape[i].value  for i in range(1, 4)])])

            prev_d = out.shape[-1].value
            for i, d in enumerate(fc_dims):
                # out = dense(out, d, 'dynamics_dense_{0}'.format(i), reuse)
                # out = tf.nn.relu(out)
                w = tf.get_variable(name='dense_w_{0}'.format(i), shape=[prev_d, d])
                b = tf.get_variable(name='dense_b_{0}'.format(i), initializer=tf.zeros([d]))
                prev_d = d
                out = tf.nn.bias_add(tf.matmul(out, w), b)
                out = tf.nn.relu(out)
                self.weights.append((w, b))

            # out = dense(out, 2*x_in.shape[-1], 'dynamics_out', reuse)
            w = tf.get_variable(name='dense_w_out'.format(i), shape=[prev_d, 2*x_in.shape[-1]])
            b = tf.get_variable(name='dense_b_out'.format(i), initializer=tf.zeros([2*x_in.shape[-1]]))
            out = tf.nn.bias_add(tf.matmul(out, w), b)
            self.weights.append((w, b))

            out_mu, out_logvar = tf.split(out, 2, axis=-1)

            if len(x_in.shape) > 2:
                out_mu = tf.reshape(out_mu, tf.shape(x_in))
                out_logvar = tf.reshape(out_logvar, tf.shape(x_in))
        return out_mu, out_logvar


    def apply(self, z_in):
        out = z_in
        for w, b in self.weights[:-1]:
            out = tf.matmul(out, w)
            out = tf.nn.bias_add(out, b)
        w, b = self.weights[-1]
        out = tf.matmul(out, w)
        out = tf.nn.bias_add(out, b)
        return out


class RecurrentLatentDynamics(object):
    def get_net(self, x_in, task_in, T, training, fc_dims=None, reuse=False, config=None):
        import tensorflow as tf

        with tf.variable_scope('latent_dynamics', reuse=reuse):
            out = tf.concat([x_in, task_in], axis=-1)
            if len(x_in.shape) > 3:
                out = tf.reshape(out, [-1, T, np.prod([out.shape[i].value  for i in range(1, 4)])])
            # self.lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(x_in.shape[-1].value, layer_norm=False, dropout_keep_prob=0., reuse=reuse)
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(x_in.shape[-1], reuse=reuse, name='lstm_cell')
            initial_state = self.lstm_cell.zero_state(batch_size=x_in.shape[0].value, dtype=tf.float32)
            self.initial_state = initial_state
            out, self.last_state = tf.nn.dynamic_rnn(self.lstm_cell, out, initial_state=initial_state,
                                           time_major=False, swap_memory=True, dtype=tf.float32,
                                           scope="RNN")
            self.weights = tf.get_variable(name='dense_w', shape=[out.shape[-1].value, 2*x_in.shape[-1]])
            self.bias = tf.get_variable(name='dense_b', initializer=tf.zeros([2*x_in.shape[-1]]))
            out = tf.nn.bias_add(tf.tensordot(out, self.weights, axes=[[2], [0]]), self.bias)
            out_mu, out_logvar = tf.split(out, 2, axis=-1)
            if len(x_in.shape) > 2:
                out_mu = tf.reshape(out_mu, tf.shape(x_in))
                out_logvar = tf.reshape(out_logvar, tf.shape(x_in))
        return out_mu, out_logvar, initial_state, self.last_state
