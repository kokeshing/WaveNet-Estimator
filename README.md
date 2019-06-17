# Wavenet

[Wavenet](https://arxiv.org/abs/1609.03499)

This repo is use tf.estimator for wavenet vocoder.
Input audio is raw or mu-law quantize.

## dependence

- tensorflow v1
- numpy
- librosa
- scipy
- sklearn
- tqdm

## mu-law generating results

https://kokeshing.com/wavenet/

## WIP

- gaussian wavenet(Implemented)
- global conditioning(Implemented)
- 2dconv, 1dconv, Nearest neighbor upsampling for local conditioning(Implemented)
- ema(Not implemented)

## Reference

- [Wavenet](https://arxiv.org/abs/1609.03499)
- [Tactron-2](https://github.com/Rayhane-mamah/Tacotron-2)
- [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- [WN-based TTSやりました](https://r9y9.github.io/blog/2018/05/20/tacotron2/)
- [Synthesize Human Speech with WaveNet](https://chainer-colab-notebook.readthedocs.io/ja/latest/notebook/official_example/wavenet.html)
- [VQ-VAEの追試で得たWaveNetのノウハウをまとめてみた。](https://www.monthly-hack.com/entry/2018/02/23/203208)
- [複数話者WaveNetボコーダに関する調査](https://www.slideshare.net/t_koshikawa/wavenet-87105461)
