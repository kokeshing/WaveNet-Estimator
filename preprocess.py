import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from audio import *
from hparams import hparams


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def audio_preprocess(wav_path, writer, mel_filter=None):
    wav = load_wav(wav_path, sr=hparams.sample_rate)

    if hparams.trim_silence:
        wav = trim_silence(wav, top_db=hparams.trim_top_db, fft_size=hparams.trim_fft_size,
                           hop_size=hparams.trim_hop_size)

    if len(wav) < hparams.max_time_steps:
        print("audio length short tha max time step")
        wav = np.pad(wav, (0, hparams.max_time_steps - len(wav)),
                     mode='constant', constant_values=0.)

    if hparams.preemphasize:
        pre_emp_wav = preemphasis(wav, hparams.preemphasis)

    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        assert (wav <= 1.).any() and (wav >= -1.).any()

        if hparams.preemphasize:
            pre_emp_wav = pre_emp_wav / np.abs(pre_emp_wav).max() * hparams.rescaling_max
            assert (pre_emp_wav <= 1.).any() and (pre_emp_wav >= -1.).any()

    if hparams.input_type == "mulaw-quantize":
        out = mulaw_quantize(wav, hparams.quantize_channels)

        start, end = start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]
        if hparams.preemphasize:
            pre_emp_wav = pre_emp_wav[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    else:
        assert hparams.input_type == "raw"
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    if hparams.preemphasize:
        mel_sp = melspectrogram(
            pre_emp_wav, n_fft=hparams.n_fft, hop_size=hparams.hop_size,
            win_size=hparams.win_size, sr=hparams.sample_rate,
            magnitude_power=hparams.magnitude_power,
            num_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax,
            min_level_db=hparams.min_level_db, ref_level_db=hparams.ref_level_db,
            mel_filter=mel_filter
        )
    else:
        mel_sp = melspectrogram(
            wav, n_fft=hparams.n_fft, hop_size=hparams.hop_size,
            win_size=hparams.win_size, sr=hparams.sample_rate,
            magnitude_power=hparams.magnitude_power,
            num_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax,
            min_level_db=hparams.min_level_db, ref_level_db=hparams.ref_level_db,
            mel_filter=mel_filter
        )

    if hparams.signal_normalization:
        mel_sp = normalize(mel_sp, max_abs_value=hparams.max_abs_value,
                           min_level_db=hparams.min_level_db, symmetric=hparams.symmetric_mels,
                           clipping=hparams.allow_clipping_in_normalization)

    mel_sp = mel_sp.astype(np.float32)
    mel_sp_len = mel_sp.shape[1]

    pad = (wav.shape[0] // hparams.hop_size + 1) * hparams.hop_size - wav.shape[0]
    out = np.pad(out, (0, pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_sp_len * hparams.hop_size

    out = out[:mel_sp_len * hparams.hop_size]
    out_len = len(out)

    assert len(out) % hparams.hop_size == 0

    out = np.array(out).astype(out_dtype)
    mel_sp = np.array(mel_sp).astype(np.float32)
    mel_sp_channels = mel_sp.shape[0]
    mel_sp_frames = mel_sp.shape[1]
    mel_sp = mel_sp.flatten()
    if out_dtype == np.float32:
        record = tf.train.Example(features=tf.train.Features(feature={
            'wav': _float32_array_feature(out),
            'length': _int64_feature(out_len),
            'mel_sp': _float32_array_feature(mel_sp),
            'mel_sp_frames': _int64_feature(mel_sp_frames),
            'mel_sp_channels': _int64_feature(mel_sp_channels)
        }))
    else:
        record = tf.train.Example(features=tf.train.Features(feature={
            'wav': _int64_array_feature(out),
            'length': _int64_feature(out_len),
            'mel_sp': _float32_array_feature(mel_sp),
            'mel_sp_frames': _int64_feature(mel_sp_frames),
            'mel_sp_channels': _int64_feature(mel_sp_channels)
        }))

    writer.write(record.SerializeToString())


def createTFRecord(data_path):
    p = Path(data_path)
    train_eval_files, test_files = train_test_split(list(p.glob("./*/wav/*.wav")),
                                                    test_size=0.05, shuffle=True)

    train_files, eval_files = train_test_split(train_eval_files,
                                               test_size=0.05, shuffle=True)

    for file_path in test_files:
        shutil.move(str(file_path), hparams.test_file_directory)

    mel_filter = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                                     fmin=hparams.fmin, fmax=hparams.fmax)

    writer = tf.python_io.TFRecordWriter(
        hparams.train_tfrecord,
        options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    )
    for wav_path in tqdm(train_files):
        audio_preprocess(str(wav_path), writer, mel_filter=mel_filter)

    writer.close()

    writer = tf.python_io.TFRecordWriter(
        hparams.eval_tfrecord,
        options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    )
    for wav_path in tqdm(eval_files):
        audio_preprocess(str(wav_path), writer, mel_filter=mel_filter)

    writer.close()


def extract_mel_sp(wav_path):
    wav = load_wav(wav_path, sr=hparams.sample_rate)

    if hparams.trim_silence:
        wav = trim_silence(wav, top_db=hparams.trim_top_db, fft_size=hparams.trim_fft_size,
                           hop_size=hparams.trim_hop_size)

    if len(wav) < hparams.max_time_steps:
        print("audio length short tha max time step")
        wav = np.pad(wav, (0, hparams.max_time_steps - len(wav)),
                     mode='constant', constant_values=0.)

    if hparams.preemphasize:
        pre_emp_wav = preemphasis(wav, hparams.preemphasis)

    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        assert (wav <= 1.).any() and (wav >= -1.).any()

        if hparams.preemphasize:
            pre_emp_wav = pre_emp_wav / np.abs(pre_emp_wav).max() * hparams.rescaling_max
            assert (pre_emp_wav <= 1.).any() and (pre_emp_wav >= -1.).any()

    if hparams.input_type == "mulaw-quantize":
        out = mulaw_quantize(wav, hparams.quantize_channels)

        start, end = start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        if hparams.preemphasize:
            pre_emp_wav = pre_emp_wav[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    mel_filter = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                                     fmin=hparams.fmin, fmax=hparams.fmax)
    if hparams.preemphasize:
        mel_sp = melspectrogram(
            pre_emp_wav, n_fft=hparams.n_fft, hop_size=hparams.hop_size,
            win_size=hparams.win_size, sr=hparams.sample_rate,
            magnitude_power=hparams.magnitude_power,
            num_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax,
            min_level_db=hparams.min_level_db, ref_level_db=hparams.ref_level_db,
            mel_filter=mel_filter
        )
    else:
        mel_sp = melspectrogram(
            wav, n_fft=hparams.n_fft, hop_size=hparams.hop_size,
            win_size=hparams.win_size, sr=hparams.sample_rate,
            magnitude_power=hparams.magnitude_power,
            num_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax,
            min_level_db=hparams.min_level_db, ref_level_db=hparams.ref_level_db,
            mel_filter=mel_filter
        )

    if hparams.signal_normalization:
        mel_sp = normalize(mel_sp, max_abs_value=hparams.max_abs_value,
                           min_level_db=hparams.min_level_db, symmetric=hparams.symmetric_mels,
                           clipping=hparams.allow_clipping_in_normalization)

    mel_sp = np.array(mel_sp).astype(np.float32)

    return mel_sp

if __name__ == '__main__':
    createTFRecord(hparams.data_directory)
