import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    # model config
    input_type="mulaw-quantize", # or raw
    quantize_channels=256,
    use_bias=True,
    legacy=True, # skip connection * sqrt(0.5)
    residual_legacy=True,  # residual layer * sqrt(0.5)
    out_channels=256,  # raw: 2, mu-law: 256
    n_layer=10,  # 1~2^(n_layer - 1) dilations
    n_loop=2,  # n_loop * 1~2^(n_layer - 1) layers
    kernel_size=3,
    dropout=0.05,
    residual_channels=128,
    dilated_channels=256,
    skip_out_channels=128,

    # loss config
    log_scale_min=float(np.log(1e-14)),
    log_scale_min_gauss=float(np.log(1e-7)),
    cdf_loss=False,

    # local conditioning (lingstic feature)
    local_conditioning=True,
    cin_channels=80,
    upsample_type='PixelShuffler',  # learnable: 1d, 2d, PixelShuffler; no learnable: NearestNeighbor
    upsample_activation='Relu',  # Relu or None
    upsample_scales=[11, 25],  # exsample: 11 * 25 = 275(hop_size)
    freq_axis_kernel_size=3,

    # global conditioning (speaker label)
    global_conditioning=False,
    gin_channels=0,
    use_speaker_embedding=True,
    n_speakers=0,
    gin_classes=0,

    # data config
    clip_mels_length=True,
    max_mel_frames=900,
    max_time_steps=8000,
    num_freq=1025,
    rescale=True,
    rescaling_max=0.999,

    # trim silence config
    trim_silence=True,
    trim_fft_size=2048,
    trim_hop_size=512,
    trim_top_db=40,
    silence_threshold=2,

    # preemphasis
    # Lfilterによってスペクトログラムのノイズとモデルの音声レベルを向上する
    preemphasize=True,
    preemphasis=0.97,

    # Mel spectrogram config
    n_fft=2048,
    hop_size=275,
    win_size=1100,
    sample_rate=22050,
    num_mels=80,
    magnitude_power=2.,
    fmin=55,
    fmax=7600,

    # Limits
    min_level_db=-100,
    ref_level_db=20,

    # mel spectrogram normalization config
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.,

    # optimaizer, decay config
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-6,
    exponential_decay_rate=0.5,
    exponential_decay_steps=200000,
    gradient_max_norm=100.0,
    gradient_max_value=5.0,

    # train config
    num_gpus=1,
    shuffle_size=350,  # shuffle size
    batch_size=8,
    max_steps=1200000,  # max iterations
    eval_step=10,
    start_delay_secs=0,  # no eval before this sec from start
    throttle_secs=900,
    save_summary_steps=250,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=10,

    # directory config
    model_directory="./result",
    data_directory="../jsut",
    train_tfrecord="./dataset/train_data",
    eval_tfrecord="./dataset/eval_data",
    test_file_directory="../jsut/test/"
)
