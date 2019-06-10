import numpy as np
import tensorflow as tf

from wavenet.model import wavenet_fn
from audio import normalize, save_wav, inv_mulaw_quantize
from preprocess import extract_mel_sp, load_wav
from hparams import hparams


def synthesize(mel_sp, save_path):
    assert len(mel_sp.shape) == 2
    mel_sp = np.expand_dims(mel_sp, axis=0)
    assert mel_sp.shape[1] == hparams.num_mels
    max_time_frame = mel_sp.shape[2]

    audio_len = max_time_frame * hparams.hop_size

    batch = {"c": mel_sp}

    wavenet = tf.estimator.Estimator(
        model_fn=wavenet_fn,
        model_dir=hparams.model_directory,
        params={
            'feature_columns': tf.feature_column.numeric_column(key="c",
                                                  shape=[hparams.num_mels, max_time_frame],
                                                  dtype=tf.float32),
            'hparams': hparams,
            'time_len': audio_len
        }
    )

    input_fn = tf.estimator.inputs.numpy_input_fn(x=batch, batch_size=1,
                                                  shuffle=False, num_epochs=1)

    wavenet_checkpoint = wavenet.latest_checkpoint()
    wavenet_outputs = wavenet.predict(input_fn=input_fn, checkpoint_path=wavenet_checkpoint)
    for result in wavenet_outputs:
        outputs = result['outputs']

        if hparams.input_type == "mulaw-quantize":
            outputs = inv_mulaw_quantize(outputs)

        save_wav(outputs, save_path, hparams.sample_rate)


if __name__ == '__main__':
    import glob
    for audio_path in glob.glob(hparams.test_file_directory + "*.wav"):
        mel_sp = extract_mel_sp(audio_path)
        synthesize(mel_sp, audio_path[:-4] + "_synthesis.wav")
