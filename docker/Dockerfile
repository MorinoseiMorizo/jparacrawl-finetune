FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /workspace
RUN chmod -R a+w .

RUN apt-get update && apt-get install -y python3-dev python3-pip wget vim

RUN pip3 install torch torchvision

# sentencepiece
RUN apt-get -y install git cmake build-essential pkg-config libgoogle-perftools-dev && \
    git clone https://github.com/google/sentencepiece.git && \
    cd sentencepiece && mkdir build && cd build && \
    cmake ..  && make -j $(nproc) && make install && \
    ldconfig -v
WORKDIR /workspace

# fairseq
RUN git clone https://github.com/pytorch/fairseq && cd fairseq && \
    git checkout c81fed46ac7868c6d80206ff71c6f6cfe93aee22 && \
    pip3 install --editable .


WORKDIR /workspace

# sacrebleu
RUN pip3 install sacrebleu

# mecab for evaluation
RUN apt-get -y install mecab libmecab-dev mecab-ipadic-utf8

# accept Japanese input
RUN apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG=ja_JP.UTF-8
