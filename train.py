import tensorflow as tf

from wavenet.model import wavenet_fn
from dataset import create_input_fn
from hparams import hparams


def train():
    if hparams.max_time_steps % hparams.hop_size == 0:
        max_steps = hparams.max_time_steps
    else:
        max_steps = hparams.max_time_steps - hparams.max_time_steps % hparams.hop_size
    max_time_frames = max_steps // hparams.hop_size

    train_input_fn, eval_input_fn = create_input_fn()
    # Create Estimator config
    if hparams.num_gpus > 1:
        conf = tf.estimator.RunConfig(
            save_summary_steps=hparams.save_summary_steps,
            save_checkpoints_steps=hparams.save_checkpoints_steps,
            keep_checkpoint_max=hparams.keep_checkpoint_max,
            train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=hparams.num_gpus)
        )
    elif hparams.num_gpus == 1:
        gpu_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0"))
        conf = tf.estimator.RunConfig(
            save_summary_steps=hparams.save_summary_steps,
            save_checkpoints_steps=hparams.save_checkpoints_steps,
            keep_checkpoint_max=hparams.keep_checkpoint_max,
            session_config=gpu_conf
        )
    else:
        conf = tf.estimator.RunConfig(
            save_summary_steps=hparams.save_summary_steps,
            save_checkpoints_steps=hparams.save_checkpoints_steps,
            keep_checkpoint_max=hparams.keep_checkpoint_max,
        )

    # Create Estimator
    if hparams.input_type == "mulaw-quantize":
        dtype = tf.int64
    else:
        assert hparams.input_type == "raw"
        dtype = tf.float32

    wavenet = tf.estimator.Estimator(
        model_fn=wavenet_fn,
        model_dir=hparams.model_directory,
        config=conf,
        params={
            'feature_columns':
                [tf.feature_column.numeric_column(key="x",
                                                  shape=[max_steps],
                                                  dtype=dtype),
                 tf.feature_column.numeric_column(key="mask",
                                                  shape=[max_steps],
                                                  dtype=tf.float32),
                 tf.feature_column.numeric_column(key="c",
                                                  shape=[hparams.num_mels, max_time_frames],
                                                  dtype=tf.float32)],
            'hparams': hparams,
            'learning_rate': hparams.learning_rate
        }
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=hparams.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.eval_step,
                                      start_delay_secs=hparams.start_delay_secs,
                                      throttle_secs=hparams.throttle_secs)
    # Run training and evaluation
    tf.estimator.train_and_evaluate(wavenet, train_spec, eval_spec)


if __name__ == '__main__':
    train()
