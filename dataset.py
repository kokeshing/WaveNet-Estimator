import tensorflow as tf
import multiprocessing

from hparams import hparams


def parse_function(example_proto):
    if hparams.input_type == "mulaw-quantize":
        dtype = tf.int64
    else:
        dtype = tf.float32

    features = {
        'wav': tf.FixedLenSequenceFeature([], dtype, allow_missing=True),
        'length': tf.FixedLenFeature([], tf.int64),
        'mel_sp': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'mel_sp_frames': tf.FixedLenFeature([], tf.int64),
        'mel_sp_channels': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    mel_sp_frames = tf.cast(parsed_features['mel_sp_frames'], tf.int32)
    mel_sp_channels = tf.cast(parsed_features['mel_sp_channels'], tf.int32)
    mel_sp = tf.reshape(parsed_features['mel_sp'], tf.stack([mel_sp_channels, mel_sp_frames]))

    return parsed_features['wav'], parsed_features['length'], mel_sp


def get_mask(inputs, max_len=None, expand=False):
    if max_len is None:
        max_len = tf.reduce_max(inputs)

    if expand:
        return tf.expand_dims(tf.sequence_mask(inputs, max_len, dtype=tf.float32), axis=-1)

    return tf.sequence_mask(inputs, max_len, dtype=tf.float32)


def adjust_time_resolution(wav, length, mel_sp):
    if hparams.max_time_steps % hparams.hop_size == 0:
        max_steps = hparams.max_time_steps
    else:
        max_steps = hparams.max_time_steps - hparams.max_time_steps % hparams.hop_size

    max_time_frames = max_steps // hparams.hop_size

    start = tf.random.uniform([1], minval=0, maxval=tf.shape(mel_sp)[0] - max_time_frames,
                              dtype=tf.int32)[0]
    time_start = start * hparams.hop_size

    inputs = wav[time_start:time_start + max_time_frames * hparams.hop_size]
    targets = tf.identity(inputs, name="targets")
    mel_sp = mel_sp[:, start:start + max_time_frames]

    mask = get_mask(length, max_len=max_steps)

    return {"x": {"x": inputs}, "c": {"c": mel_sp}, "mask": {"mask": mask}}, targets


def create_input_fn():
    def _train_input_fn():
        num_threads = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(filenames=hparams.train_tfrecord, compression_type='GZIP')\
            .apply(tf.contrib.data.shuffle_and_repeat(hparams.shuffle_size, None))\
            .map(parse_function, num_parallel_calls=num_threads)\
            .map(adjust_time_resolution, num_parallel_calls=num_threads)\
            .batch(hparams.batch_size)\
            .prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def _eval_input_fn():
        num_threads = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(filenames=hparams.eval_tfrecord, compression_type='GZIP')\
            .map(parse_function, num_parallel_calls=num_threads)\
            .map(adjust_time_resolution, num_parallel_calls=num_threads)\
            .apply(tf.contrib.data.shuffle_and_repeat(hparams.shuffle_size, 1))\
            .batch(1)\
            .prefetch(tf.contrib.data.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    return _train_input_fn, _eval_input_fn
