import tensorflow as tf
import numpy as np

from .module import *
from audio import mulaw_quantize


class WaveNet(tf.keras.Model):
    def __init__(self, hparams):
        super().__init__()

        self._hparams = hparams

        if hparams.local_conditioning:
            assert hparams.num_mels == hparams.cin_channels

        with tf.variable_scope('input_convolution'):
            self.first_layer = CasualConv1D(hparams.residual_channels,
                                            name='input_convolution')

        self.residual_layers = []
        for loop_index in range(hparams.n_loop):
            for i in range(hparams.n_layer):
                layer_index = i + loop_index * hparams.n_layer
                self.residual_layers.append(
                    ResidualConv1DGLU(hparams.residual_channels,
                                      hparams.dilated_channels,
                                      kernel_size=hparams.kernel_size,
                                      skip_out_channels=hparams.skip_out_channels,
                                      use_bias=hparams.use_bias,
                                      dilation_rate=2 ** i,
                                      dropout=hparams.dropout,
                                      cin_channels=hparams.cin_channels,
                                      gin_channels=hparams.gin_channels,
                                      residual_legacy=hparams.residual_legacy,
                                      name='ResidualConv1DGLU_{}'.format(layer_index + 1))
                )

        with tf.variable_scope('final_layers'):
            self.final_layers = [
                ReluActivation(name='final_layer_relu1'),
                CasualConv1D(hparams.skip_out_channels,
                             name='final_convolution_1'),
                ReluActivation(name='final_layer_relu2'),
                CasualConv1D(hparams.out_channels,
                             name='final_convolution_2')
            ]

        self.all_layers = [self.first_layer] + self.residual_layers + self.final_layers

        # global conditioning
        if hparams.global_conditioning:
            assert hparams.n_speakers > 0
            self.embed_speakers = Embedding(hparams.n_speakers, hparams.gin_channels)

        else:
            self.embed_speakers = None

        # local conditioning  inputs: [batch_size, freq, time_steps, channels]
        if hparams.local_conditioning:
            self.upsample_layers = []
            if hparams.upsample_type == 'NearestNeighbor':
                self.upsample_layers.append(
                    NearestNeighborUpsample(strides=(1, hparams.hop_size))
                )

            else:
                for i, s in enumerate(hparams.upsample_scales):
                    with tf.variable_scope('local_conditioning_upsampling_{}'.format(i + 1)):
                        if hparams.upsample_type == '2d':
                            conv = tf.layers.Conv2DTranspose(
                                filters=1,
                                kernel_size=(hparams.freq_axis_kernel_size, s),
                                strides=(1, s),
                                padding='same',
                                kernel_initializer=None,
                                bias_initializer=tf.zeros_initializer(),
                                data_format='channels_last',
                                name='Conv2dTranspose_layer_{}'.format(i + 1))

                        elif hparams.upsample_type == '1d':
                            conv = tf.layers.Conv1dTranspose(
                                filter=hparams.cin_channels,
                                kernel_size=(s,),
                                strides=(s,),
                                padding='same',
                                kernel_initializer=None,
                                bias_initializer=tf.zeros_initializer(),
                                data_format='channels_last',
                                name='Conv2dTranspose_layer_{}'.format(i + 1))

                        elif hparams.upsample_type == 'PixelShuffler':
                            conv = PixelShuffler(
                                1,
                                (hparams.freq_axis_kernel_size, 3),
                                padding='same', strides=(1, s),
                                kernel_initializer=None,
                                name='PixelShuffler_{}'.format(i + 1))

                        else:
                            raise AttributeError("{} not found".format(hparams.upsample_type))

                        self.upsample_layers.append(conv)
                        self.upsample_layers.append(
                            ReluActivation(name='upsample_relu_{}'.format(i + 1))
                        )

            self.all_layers = self.upsample_layers + self.all_layers

    def call(self, inputs, g=None, c=None):
        """
            build calicurate graph

            inputs : [batch_size, time_len, channels] audio signal
            g: [batch_size, gin_channels] global conditioning feature (ex. speaker label)
            c: [batch_size, time_len, cin_channels] local conditioning feature
               (ex. linguistic feature)
        """
        batch_size = tf.shape(inputs)[0]
        time_len = tf.shape(inputs)[1]

        # global conditioning
        if g is not None:
            g = tf.expand_dims(g, -1)
            if self.embed_speakers is not None:
                g = self.embed_speakers(g)

            with tf.control_dependencies([tf.assert_equal(tf.shape(g)[0], batch_size)]):
                g = tf.reshape(g, [tf.shape(g)[0], 1, tf.shape(g)[1]])

            g = tf.tile(g, [1, time_len, 1])

        # local conditioning
        if c is not None:
            if self._hparams.upsample_type == '2d':
                expand_dim = 1
            elif self._hparams.upsample_type == '1d':
                expand_dim = 2
            else:
                assert self._hparams.upsample_type in ('PixelShuffler', 'NearestNeighbor')
                expand_dim = 3

            c = tf.expand_dims(c, axis=expand_dim)

            for upsample_layer in self.upsample_layers:
                c = upsample_layer(c)

            c = tf.squeeze(c, [expand_dim])
            with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], time_len)]):
                c = tf.transpose(c, [0, 2, 1])

        # feed
        x = self.first_layer(inputs)
        skips = None
        for layer in self.residual_layers:
            x, h = layer(x, c=c, g=g)
            if skips is None:
                skips = h
            else:
                skips = skips + h
                if self._hparams.legacy:
                    skips = skips * np.sqrt(0.5)

        x = skips
        for layer in self.final_layers:
            x = layer(x)

        return x

    def incremental_feed(self, initial_inputs, test_inputs=None, g=None, c=None, time_len=100,
                         softmax=False, quantize=True, synthesis=False,
                         log_scale_min=-7.0,
                         log_scale_min_gauss=-7.0):
        batch_size = tf.shape(initial_inputs)[0]

        if test_inputs is not None:
            batch_size = tf.shape(test_inputs)[0]
            if self._hparams.input_type == "raw":
                if tf.shape(test_inputs)[1] == self._hparams.out_channels:
                    test_inputs = tf.transpose(test_inputs, [0, 2, 1])

            else:
                test_inputs = tf.cast(test_inputs, tf.int32)
                test_inputs = tf.one_hot(indices=test_inputs,
                                         depth=self._hparams.quantize_channels,
                                         dtype=tf.float32)
                test_inputs = tf.squeeze(test_inputs, [2])

                if tf.shape(test_inputs)[1] == self._hparams.out_channels:
                    test_inputs = tf.transpose(test_inputs, [0, 2, 1])

            if time_len is None:
                time_len = tf.shape(test_inputs)[1]
            else:
                time_len = tf.maximum(time_len, tf.shape(test_inputs)[1])

        # global conditioning
        if g is not None:
            g = tf.expand_dims(g, -1)
            if self.embed_speakers is not None:
                g = self.embed_speakers(g)

            with tf.control_dependencies([tf.assert_equal(tf.shape(g)[0], batch_size)]):
                # [batch_size, 1, gin_channels]
                g = tf.reshape(g, [tf.shape(g)[0], 1, tf.shape(g)[1]])

            # adjast len of dim of time [speaker_id] => [speaker_id, ..., speaker_id]
            self.g = tf.tile(g, [1, time_len, 1])

        # local conditioning
        if c is not None:
            if self._hparams.upsample_type == '2d':
                expand_dim = 1
            elif self._hparams.upsample_type == '1d':
                expand_dim = 2
            else:
                assert self._hparams.upsample_type in ('PixelShuffler', 'NearestNeighbor')
                expand_dim = 3

            c = tf.expand_dims(c, axis=expand_dim)

            for upsample_layer in self.upsample_layers:
                c = upsample_layer(c)

            c = tf.squeeze(c, [expand_dim])
            with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], time_len)]):
                self.c = tf.transpose(c, [0, 2, 1])

        # prepare for tf.while_loop
        initial_time = tf.constant(0, dtype=tf.int32)
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        initial_loss_outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        initial_queues = [tf.zeros((batch_size,
                                    r_layer.layer.kw + (r_layer.layer.kw - 1) * (r_layer.layer.dilation_rate - 1),
                                    r_layer.residual_channels),
                                    name='queue_{}'.format(i + 1)) for i, r_layer in enumerate(self.residual_layers)]

        def cond(time, _outputs, _current_input, _loss_outputs, _queues):
            return tf.less(time, time_len)

        def body(time, outputs, inputs, loss_outputs, queues):
            if self._hparams.local_conditioning:
                cf = tf.expand_dims(self.c[:, time, :], axis=1)
            else:
                cf = None

            if self._hparams.global_conditioning:
                gf = tf.expand_dims(self.g[:, time, :], axis=1)
            else:
                gf = None

            x = self.first_layer(inputs, is_incremental=True)

            skips = None
            new_queues = []
            for layer, queue in zip(self.residual_layers, queues):
                x, h, new_queue = layer(x, c=cf, g=gf, is_incremental=True, queue=queue)

                new_queues.append(new_queue)
                if self._hparams.legacy:
                    if skips is not None:
                        skips = (skips + h) * np.sqrt(0.5)
                    else:
                        skips = h
                else:
                    if skips is not None:
                        skips = skips + h
                    else:
                        skips = h

            x = skips
            for layer in self.final_layers:
                x = layer(x, is_incremental=True)

            loss_outputs = loss_outputs.write(time, tf.squeeze(x, [1]))

            if self._hparams.input_type == "raw":
                assert self._hparams.out_channels == 2
                x = SampleFromGaussian(tf.reshape(x, [batch_size, 1, -1]),
                                       log_scale_min_gauss=log_scale_min_gauss)

                next_inputs = tf.expand_dims(x, axis=-1)

            else:
                if softmax:
                    x = tf.nn.softmax(x, axis=1)
                else:
                    x = tf.reshape(x, [batch_size, -1])

                if quantize:
                    sample = tf.multinomial(x, 1)
                    x = tf.one_hot(sample, depth=self._hparams.quantize_channels)

                next_inputs = x

            if len(x.shape) == 3:
                x = tf.squeeze(x, [1])

            outputs = outputs.write(time, x)

            if test_inputs is not None:
                next_inputs = tf.expand_dims(test_inputs[:, time, :], axis=1)

            time = time + 1

            return time, outputs, next_inputs, loss_outputs, new_queues

        result = tf.while_loop(
            cond,
            body,
            loop_vars=[initial_time, initial_outputs, initial_inputs, initial_loss_outputs, initial_queues],
            parallel_iterations=32,
            swap_memory=False
        )

        # outputs [time, batch, channel]
        outputs = result[1].stack()
        eval_outputs = result[3].stack()

        if synthesis:
            return tf.transpose(outputs, [1, 0, 2])
        else:
            return tf.transpose(eval_outputs, [1, 0, 2])


def wavenet_fn(features, labels, mode, params):
    # input features
    feature_columns = params['feature_columns']
    hparams = params['hparams']

    # build model
    wavenet = WaveNet(hparams)

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        inputs = tf.feature_column.input_layer(features["x"], feature_columns[0])
        max_time_len = feature_columns[0].shape[0]
        mask = tf.feature_column.input_layer(features["mask"], feature_columns[1])
        if hparams.local_conditioning:
            c = tf.feature_column.input_layer(features["c"], feature_columns[2])
            num_mels = feature_columns[2].shape[0]
            max_time_frames = feature_columns[2].shape[1]

        batch_size = tf.shape(inputs)[0]
        labels = tf.reshape(labels, [batch_size, max_time_len])
        mask = tf.reshape(mask, [batch_size, max_time_len])
        if hparams.input_type == "mulaw-quantize":
            inputs = tf.cast(inputs, tf.int64)
            inputs = tf.one_hot(inputs, hparams.quantize_channels, axis=-1, dtype=tf.float32)
            labels = tf.expand_dims(labels, axis=-1)
            mask = mask[:, 1:]

        else:
            inputs = tf.expand_dims(inputs, axis=-1)
            labels = tf.expand_dims(labels, axis=-1)
            mask = tf.expand_dims(mask, axis=-1)[:, 1:, :]

        if hparams.local_conditioning:
            c = tf.reshape(c, [batch_size, num_mels, max_time_frames])

        outputs = wavenet(inputs, g=None, c=c)

        if hparams.input_type == "raw":
            loss = GaussianMaximumLikelihoodEstimationLoss(outputs[:, :-1, :], labels[:, 1:, :],
                                                           hparams.log_scale_min_gauss,
                                                           hparams.quantize_channels,
                                                           use_cdf=hparams.cdf_loss,
                                                           reduce=False, mask=mask)
        else:
            assert hparams.input_type == "mulaw-quantize"
            loss = MaskedSoftMaxCrossEntropyLoss(outputs[:, :-1, :], labels[:, 1:], mask=mask)

        tf.summary.scalar('loss', loss)

        learning_rate = float(params['learning_rate'])
        learning_rate = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(),
                                                   hparams.exponential_decay_steps,
                                                   hparams.exponential_decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate, hparams.beta1,
                                           hparams.beta2, hparams.epsilon)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, hparams.gradient_max_norm)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        inputs = tf.feature_column.input_layer(features["x"], feature_columns[0])
        max_time_len = feature_columns[0].shape[0]
        mask = tf.feature_column.input_layer(features["mask"], feature_columns[1])
        if hparams.local_conditioning:
            c = tf.feature_column.input_layer(features["c"], feature_columns[2])
            num_mels = feature_columns[2].shape[0]
            max_time_frames = feature_columns[2].shape[1]

        labels = tf.reshape(labels, [1, -1])
        mask = tf.reshape(mask, [1, max_time_len])
        if hparams.input_type == "mulaw-quantize":
            labels = tf.expand_dims(labels, axis=-1)

        else:
            labels = tf.expand_dims(labels, axis=-1)
            mask = tf.expand_dims(mask, axis=-1)

        if hparams.local_conditioning:
            c = tf.reshape(c, [1, num_mels, max_time_frames])

        test_inputs = tf.reshape(labels, [1, -1, 1])

        if hparams.input_type == "mulaw-quantize":
            initial_value = mulaw_quantize(0, hparams.quantize_channels)
        else:
            initial_value = 0.0

        if hparams.input_type == "mulaw-quantize":
            initial_inputs = tf.one_hot(indices=initial_value,
                                        depth=hparams.quantize_channels, dtype=tf.float32)
            initial_inputs = tf.tile(tf.reshape(initial_inputs, [1, 1, hparams.quantize_channels]),
                                     [1, 1, 1])
        else:
            initial_inputs = tf.ones([1, 1, 1], tf.float32) * initial_value

        outputs = wavenet.incremental_feed(initial_inputs, test_inputs=test_inputs, g=None, c=c,
                                           time_len=max_time_len, softmax=False, quantize=True,
                                           synthesis=False, log_scale_min=hparams.log_scale_min,
                                           log_scale_min_gauss=hparams.log_scale_min_gauss)

        if hparams.input_type == "raw":
            loss = GaussianMaximumLikelihoodEstimationLoss(outputs,
                                                           labels,
                                                           hparams.log_scale_min_gauss,
                                                           hparams.quantize_channels,
                                                           use_cdf=hparams.cdf_loss,
                                                           reduce=False, mask=mask)
        else:
            assert hparams.input_type == "mulaw-quantize"
            loss = MaskedSoftMaxCrossEntropyLoss(outputs, labels, mask=mask)

        return tf.estimator.EstimatorSpec(mode, loss=loss)

    else:
        assert mode == tf.estimator.ModeKeys.PREDICT
        if hparams.local_conditioning:
            c = tf.feature_column.input_layer(features, feature_columns)
            num_mels = feature_columns.shape[0]
            max_time_frames = feature_columns.shape[1]


        if hparams.local_conditioning:
            c = tf.reshape(c, [1, num_mels, max_time_frames])

        if hparams.input_type == "mulaw-quantize":
            initial_value = mulaw_quantize(0, hparams.quantize_channels)
        else:
            initial_value = 0.0

        batch_size = tf.shape(c)[0]
        if hparams.input_type == "mulaw-quantize":
            initial_inputs = tf.one_hot(indices=initial_value,
                                        depth=hparams.quantize_channels, dtype=tf.float32)
            initial_inputs = tf.tile(tf.reshape(initial_inputs, [1, 1, hparams.quantize_channels]),
                                     [batch_size, 1, 1])
            softmax = False
        else:
            initial_inputs = tf.ones([batch_size, 1, 1], tf.float32) * initial_value
            softmax = False

        time_len = int(params['time_len'])
        outputs = wavenet.incremental_feed(initial_inputs, test_inputs=None, g=None, c=c,
                                           time_len=time_len, softmax=softmax, quantize=True,
                                           synthesis=True, log_scale_min=hparams.log_scale_min,
                                           log_scale_min_gauss=hparams.log_scale_min_gauss)

        if hparams.input_type == "mulaw-quantize":
            outputs = tf.reshape(tf.argmax(outputs, axis=2), [batch_size, -1])
        else:
            outputs = tf.reshape(outputs, [batch_size, -1])

        predictions = {
            'outputs': outputs,
        }
        export_outputs = {
            'outputs': tf.estimator.export.PredictOutput(outputs)
        }

        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)
