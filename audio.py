import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile


def load_wav(path, sr):
    wav = librosa.core.load(path, sr=sr)[0]

    return wav.astype(np.float32)


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def mulaw(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def inv_mulaw(x, mu=255):
    return np.sign(x) * (1.0 / mu) * ((1.0 + mu) ** np.abs(x) - 1.0)


def mulaw_quantize(x, mu=255):
    x = mulaw(x)
    x = (x + 1) / 2 * mu

    return x.astype(np.int)


def inv_mulaw_quantize(x, mu=255):
    x = 2 * x.astype(np.float32) / mu - 1

    return inv_mulaw(x, mu)


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def trim_silence(wav, top_db=40, fft_size=2048, hop_size=512):
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]


def melspectrogram(wav, n_fft=2048, hop_size=275, win_size=1100, sr=22050, magnitude_power=2.,
                   num_mels=80, fmin=55, fmax=7600, min_level_db=-100, ref_level_db=20, mel_filter=None):
    d = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size,
                     win_length=win_size, pad_mode='constant')
    mel_sp = _linear_to_mel(np.abs(d) ** magnitude_power, sr, n_fft, num_mels, fmin, fmax, mel_filter=mel_filter)
    mel_sp = _amp_to_db(mel_sp, min_level_db) - ref_level_db

    return mel_sp


def _linear_to_mel(spectogram, sr, n_fft, num_mels, fmin, fmax, mel_filter=None):
    if mel_filter is None:
        mel_filter = librosa.filters.mel(sr, n_fft, n_mels=num_mels,
                                         fmin=fmin, fmax=fmax)

    return np.dot(mel_filter, spectogram)


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))

    return 20 * np.log10(np.maximum(min_level, x))


def normalize(s, max_abs_value=4., min_level_db=-100, symmetric=True, clipping=True):
    if clipping:
        if symmetric:
            s = (2 * max_abs_value) * ((s - min_level_db) / (-min_level_db)) - max_abs_value
            return np.clip(s, -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((s - min_level_db) / (-min_level_db)), 0, max_abs_value)

    if symmetric:
        return (2 * max_abs_value) * ((s - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((s - min_level_db) / (-min_level_db))
