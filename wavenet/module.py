import tensorflow as tf
import numpy as np


class Embedding:
    def __init__(self, num_embeddings, embedding_dim, std=0.1, name='gin_embedding'):
        initializer = tf.truncated_normal_initializer(mean=0., stddev=std)
        self.embedding_table = tf.get_variable(name,
                                               [num_embeddings, embedding_dim],
                                               dtype=tf.float32,
                                               initializer=initializer)

    def __call__(self, inputs, is_incremental=False):
        return tf.nn.embedding_lookup(self.embedding_table, inputs)


class ReluActivation:
    def __init__(self, name=None):
        self.name = name

    def __call__(self, inputs, is_incremental=False):
        return tf.nn.relu(inputs, name=self.name)


class CasualConv1D(tf.keras.layers.Wrapper):
    def __init__(self, filters, kernel_size=1, strides=1, data_format='channels_last',
                 dilation_rate=1, activation=None, use_bias=True, kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(), trainable=True, name=None, **kwargs
                 ):

        layer = tf.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=trainable,
            name=name, **kwargs
        )

        super().__init__(layer, name=name, **kwargs)
        self._track_trackable(layer, name='layer')
        self.filters = filters
        self.kw = kernel_size
        self.dilation_rate = dilation_rate

        self.scope = 'CausalConv1D' if name is None else name

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.layers.InputSpec(shape=input_shape)

        self.layer.build(input_shape)

        in_channels = input_shape[-1]
        self.linearized_weights = self._get_linearized_weight(in_channels)

        super().build()

    def call(self, inputs, is_incremental=False, queue=None):
        with tf.variable_scope(self.scope) as scope:
            # nomal run
            if not is_incremental:
                padding_size = (self.kw - 1) * self.dilation_rate

                # Casual convolutionのためpaddingは時系列の初めの方のみにする
                if self.layer.data_format == 'channels_last':
                    inputs_ = tf.pad(inputs, tf.constant([(0, 0), (padding_size, 0), (0, 0)]))
                else:
                    assert self.layer.data_format == 'channels_first'
                    inputs_ = tf.pad(inputs, tf.constant([(0, 0), (0, 0), (padding_size, 0)]))

                outputs = self.layer(inputs_)

                return outputs

            # incremental run
            batch_size = tf.shape(inputs)[0]
            if self.kw > 1:
                queue = queue[:, 1:, :]
                queue = tf.concat([queue, tf.expand_dims(inputs[:, -1, :], axis=1)], axis=1)

                if self.dilation_rate > 1:
                    inputs = queue[:, 0::self.dilation_rate, :]
                else:
                    inputs = queue

            outputs = tf.matmul(tf.reshape(inputs, [batch_size, -1]), self.linearized_weights)
            if self.layer.use_bias:
                outputs = tf.nn.bias_add(outputs, self.layer.bias)

            # [batch_size, 1(time_len), channels]
            if queue is None:
                return tf.reshape(outputs, [batch_size, 1, self.layer.filters])
            else:
                return tf.reshape(outputs, [batch_size, 1, self.layer.filters]), queue

    def _get_linearized_weight(self, in_channels):
        if tf.shape(self.layer.kernel) == (self.layer.filters, in_channels, self.kw):
            # [filters, in, kw]
            weight = tf.transpose(self.layer.kernel, [2, 1, 0])
        else:
            # [kw, in, filters]
            weight = self.layer.kernel

        # [kw, in, filters]
        assert weight.shape == (self.kw, in_channels, self.layer.filters)
        self.in_channels = in_channels

        return tf.cast(tf.reshape(weight, [-1, self.layer.filters]), dtype=tf.float32)


class ResidualConv1DGLU(tf.keras.layers.Wrapper):
    """
        conv1d + GLU => add condition => residual add + skip connection
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None, cin_channels=0, gin_channels=0, dropout=0.05,
                 dilation_rate=1, use_bias=True, residual_legacy=True,
                 name='ResidualConv1DGLU', **kwargs):

        self.scope = name
        self.dropout = dropout
        self.residual_channels = residual_channels
        self.residual_legacy = residual_legacy

        if skip_out_channels is None:
            skip_out_channels = residual_channels

        conv1d = CasualConv1D(gate_channels, kernel_size=kernel_size,
                              dilation_rate=dilation_rate, use_bias=use_bias,
                              name='Conv1D_{}'.format(name))

        # Local conditioning (linguistic features)
        if cin_channels:
            self.conv1d_c = CasualConv1D(gate_channels, use_bias=use_bias,
                                         name='Local_conditioning_{}'.format(name))

        # Global conditioning (Speaker label)
        if gin_channels:
            self.conv1d_g = CasualConv1D(gate_channels, use_bias=use_bias,
                                         name='Global_conditioning_{}'.format(name))
        else:
            self.conv1d_g = None

        self.conv1d_out = CasualConv1D(residual_channels, use_bias=use_bias,
                                       name='Conv1D_out_{}'.format(name))

        self.conv1d_skip = CasualConv1D(skip_out_channels, use_bias=use_bias,
                                        name='Conv1D_skip_connection_{}'.format(name))

        # tf.keras.layers.Wrapper __init__() will substitution conv1d for self.layer
        super().__init__(conv1d, name=name, **kwargs)

    def call(self, inputs, c=None, g=None, is_incremental=False, queue=None):
        with tf.variable_scope(self.scope) as scope:
            x = tf.layers.dropout(inputs, rate=self.dropout)

            if is_incremental:
                x, queue = self.layer(x, is_incremental=True, queue=queue)
            else:
                x = self.layer(x, is_incremental=False)

            # GLU
            x_tanh, x_sigmoid = tf.split(x, num_or_size_splits=2, axis=2)

            if c is not None:
                c = self.conv1d_c(c, is_incremental)
                c_tanh, c_sigmoid = tf.split(c, num_or_size_splits=2, axis=2)
                x_tanh, x_sigmoid = x_tanh + c_tanh, x_sigmoid + c_sigmoid

            if g is not None:
                g = self.conv1d_g(g, is_incremental)
                g_tanh, g_sigmoid = tf.split(g, num_or_size_splits=2, axis=2)
                x_tanh, x_sigmoid = x_tanh + g_tanh, x_sigmoid + g_sigmoid

            x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigmoid)

            s = self.conv1d_skip(x, is_incremental)
            x = self.conv1d_out(x, is_incremental)

            if self.residual_legacy:
                with tf.control_dependencies([tf.assert_equal(tf.shape(inputs)[1], tf.shape(x)[1])]):
                    x = (x + inputs) * tf.math.sqrt(0.5)
            else:
                with tf.control_dependencies([tf.assert_equal(tf.shape(inputs)[1], tf.shape(x)[1])]):
                    x = (x + inputs)

            if is_incremental:
                return x, s, queue
            else:
                return x, s


class PixelShuffler(tf.layers.Conv2D):
    def __init__(self, filters, kernel_size, strides, padding, kernel_initializer,
                 bias_initializer=tf.zeros_initializer(), data_format='channels_last',
                 name='PixelShuffler', **kwargs):
        pre_conv_filters = filters * strides[0] * strides[1]

        super().__init__(filters=pre_conv_filters, kernel_size=kernel_size,
                         strides=(1, 1), padding=padding,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, data_format=data_format,
                         name=name, **kwargs)

        self.ps_filters = filters
        self.ps_strides = strides
        self.scope = name

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        with tf.variable_scope(self.scope) as scope:
            x = super().call(inputs)

            return self.pixel_shuffler(x)

    def pixel_shuffler(self, inputs):
        batch_size = tf.shape(inputs)[0]
        _, H, W, C = inputs.get_shape()
        r1, r2 = self.ps_strides
        out_c = self.ps_filters
        out_h = H * r1
        out_w = W * r2

        assert C == r1 * r2 * out_c

        x = tf.reshape(inputs, (batch_size, H, W, r1, r2, out_c))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (batch_size, out_h, out_w, out_c))

        return x


def MaskedSoftMaxCrossEntropyLoss(outputs, targets, mask):
    # one hot encoding
    targets_ = tf.one_hot(targets, depth=tf.shape(outputs)[-1])

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets_)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(loss))]):
        loss = loss * mask

    return tf.reduce_sum(loss) / tf.count_nonzero(loss, dtype=tf.float32)


def GaussianMaximumLikelihoodEstimationLoss(outputs, targets, log_scale_min_gauss, num_classes,
                                            use_cdf=True, reduce=False, mask=None):
    assert mask is not None

    ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
    mask_ = mask * ones

    mean = outputs[:, :, 0]
    log_scale = tf.maximum(outputs[:, :, 1], log_scale_min_gauss)
    targets = tf.squeeze(targets, [-1])

    if use_cdf:
        gaussian = tf.contrib.distributions.Normal(loc=mean, scale=tf.exp(log_scale))

        cdf_plus = gaussian.cdf(targets + 1. / (num_classes - 1))
        cdf_min = gaussian.cdf(targets - 1. / (num_classes - 1))

        log_prob = tf.log(tf.maximum(cdf_plus - cdf_min, 1e-12))

    else:
        log_prob = -0.5 * (np.log(2. * np.pi) + 2. * log_scale + tf.square(targets - mean) * tf.exp(-2. * log_scale))

    if reduce:
        loss = -tf.reduce_sum(log_prob)
    else:
        loss = -tf.expand_dims(log_prob, [-1])

    return tf.reduce_sum(loss * mask_) / tf.reduce_sum(mask_)


def SampleFromGaussian(y, log_scale_min_gauss):
    mean = y[:, :, 0]
    log_scale = tf.maximum(y[:, :, 1], log_scale_min_gauss)
    scale = tf.exp(log_scale)

    gaussian_dist = tf.contrib.distributions.Normal(loc=mean, scale=scale, allow_nan_stats=False)
    x = gaussian_dist.sample()

    return tf.minimum(tf.maximum(x, -1.), 1.)
