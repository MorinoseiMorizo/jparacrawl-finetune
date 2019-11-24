#!/bin/sh

# download pre-trained model
wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/1.0/pretrained_models/ja-en/base.tar.gz
tar xzvf base.tar.gz
rm base.tar.gz
mv base ./pretrained_model_jaen
