#!/bin/bash
export python3='ignore:semaphore_tracker:UserWarning'

FAIRSEQ=/workspace/fairseq

EXP_NAME=decode

SRC=ja
TRG=en

TEST_SRC=$PWD/corpus/spm/kyoto-test.$SRC
TEST_TRG=$PWD/corpus/spm/kyoto-test.$TRG
TEST_TRG_RAW=$PWD/corpus/kftt-data-1.0/data/orig/kyoto-test.$TRG

SRC_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$SRC.txt
TRG_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$TRG.txt
SPM_MODEL=$PWD/corpus/enja_spm_models/spm.$TRG.nopretok.model

MODEL_FILE=$PWD/pretrained_model_$SRC$TRG/base.pretrain.pt

CORPUS_DIR=$PWD/data
DATA_DIR=$PWD/data-bin/$EXP_NAME
OUT_DIR=$PWD/decode

TEST_PREFIX=$CORPUS_DIR/$EXP_NAME/test

mkdir -p $CORPUS_DIR
mkdir -p $DATA_DIR
mkdir -p $OUT_DIR

# make links to corpus
mkdir -p $CORPUS_DIR/$EXP_NAME
ln -s $TEST_SRC $TEST_PREFIX.$SRC
ln -s $TEST_TRG $TEST_PREFIX.$TRG

######################################
# Preprocessing
######################################
python3 $FAIRSEQ/preprocess.py \
    --source-lang $SRC \
    --target-lang $TRG \
    --testpref $TEST_PREFIX \
    --destdir $DATA_DIR \
    --srcdict $SRC_VOCAB \
    --tgtdict $TRG_VOCAB \
    --workers `nproc` \


######################################
# Generate
######################################
# decode
B=`basename $TEST_SRC`

python3 $FAIRSEQ/generate.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_FILE \
    --max-tokens 1000 \
    --beam 6 \
    --lenpen 1.0 \
    --log-format simple \
    --remove-bpe \
    | tee $OUT_DIR/$B.hyp

grep "^H" $OUT_DIR/$B.hyp | sed 's/^H-//g' | sort -n | cut -f3 > $OUT_DIR/$B.true
cat $OUT_DIR/$B.true | spm_decode --model=$SPM_MODEL --input_format=piece > $OUT_DIR/$B.true.detok


######################################
# Evaluation
######################################
cat $OUT_DIR/$B.true.detok | sacrebleu --tokenize=intl $TEST_TRG_RAW | tee -a $OUT_DIR/test.log
