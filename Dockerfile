FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update > /dev/null
RUN apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg libav-tools wget git vim > /dev/null

RUN wget -q http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
RUN unzip jsut_ver1.1.zip -d jsut > /dev/null
RUN mkdir ./jsut/test
RUN mkdir ./jsut/tfrecord

RUN mv ./jsut/jsut_ver1.1/* ./jsut/
RUN git clone https://github.com/kokeshing/Wavenet-Estimator.git
WORKDIR Wavenet-Estimator
RUN mkdir result
RUN mkdir dataset
RUN mkdir dataset/test
RUN pip install -r requirements.txt > /dev/null
RUN cd /usr/local/cuda/lib64 \
    && mv stubs/libcuda.so ./ \
    && ln -s libcuda.so libcuda.so.1 \
    && ldconfig
CMD ["/bin/bash"]
