#!/bin/sh

mkdir -p corpus
# download KFTT
wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar xzvf kftt-data-1.0.tar.gz
rm kftt-data-1.0.tar.gz
mv kftt-data-1.0 ./corpus

# download sentencepiece model
wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/1.0/spm_models/en-ja_spm.tar.gz
tar xzvf en-ja_spm.tar.gz
rm en-ja_spm.tar.gz
mv enja_spm_models ./corpus
