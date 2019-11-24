#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

FAIRSEQ=/workspace/fairseq

SEED=1

EXP_NAME=fine-tune

SRC=ja
TRG=en

TRAIN_SRC=$PWD/corpus/spm/kyoto-train.$SRC
TRAIN_TRG=$PWD/corpus/spm/kyoto-train.$TRG
DEV_SRC=$PWD/corpus/spm/kyoto-dev.$SRC
DEV_TRG=$PWD/corpus/spm/kyoto-dev.$TRG
TEST_SRC=$PWD/corpus/spm/kyoto-test.$SRC
TEST_TRG=$PWD/corpus/spm/kyoto-test.$TRG
TEST_TRG_RAW=$PWD/corpus/kftt-data-1.0/data/orig/kyoto-test.$TRG

SRC_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$SRC.txt
TRG_VOCAB=$PWD/pretrained_model_$SRC$TRG/dict.$TRG.txt
SPM_MODEL=$PWD/corpus/enja_spm_models/spm.$TRG.nopretok.model

PRETRAINED_MODEL_FILE=$PWD/pretrained_model_$SRC$TRG/base.pretrain.pt

SPM_MODEL=$PWD/corpus/enja_spm_models/spm.$TRG.nopretok.model

CORPUS_DIR=$PWD/data
MODEL_DIR=$PWD/models/$EXP_NAME
DATA_DIR=$PWD/data-bin/$EXP_NAME

TRAIN_PREFIX=$CORPUS_DIR/$EXP_NAME/train
DEV_PREFIX=$CORPUS_DIR/$EXP_NAME/dev
TEST_PREFIX=$CORPUS_DIR/$EXP_NAME/test

mkdir -p $CORPUS_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR

# make links to corpus
mkdir -p $CORPUS_DIR/$EXP_NAME
ln -s $TRAIN_SRC $TRAIN_PREFIX.$SRC
ln -s $TRAIN_TRG $TRAIN_PREFIX.$TRG
ln -s $DEV_SRC $DEV_PREFIX.$SRC
ln -s $DEV_TRG $DEV_PREFIX.$TRG
ln -s $TEST_SRC $TEST_PREFIX.$SRC
ln -s $TEST_TRG $TEST_PREFIX.$TRG

######################################
# Preprocessing
######################################
python3 $FAIRSEQ/preprocess.py \
    --source-lang $SRC \
    --target-lang $TRG \
    --trainpref $TRAIN_PREFIX \
    --validpref $DEV_PREFIX \
    --testpref $TEST_PREFIX \
    --destdir $DATA_DIR \
    --srcdict $SRC_VOCAB \
    --tgtdict $TRG_VOCAB \
    --workers `nproc` \


######################################
# Training
######################################
python3 $FAIRSEQ/train.py $DATA_DIR \
    --restore-file $PRETRAINED_MODEL_FILE \
    --arch transformer \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.001 \
    --min-lr 1e-09 \
    --dropout 0.3 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 5000 \
    --max-update 28000 \
    --save-dir $MODEL_DIR \
    --no-epoch-checkpoints \
    --save-interval 10000000000 \
    --validate-interval 1000000000 \
    --save-interval-updates 100 \
    --keep-interval-updates 8 \
    --log-format simple \
    --log-interval 5 \
    --ddp-backend no_c10d \
    --update-freq 16 \
    --fp16 \
    --seed $SEED \


######################################
# Averaging
######################################
rm -rf $MODEL_DIR/average
mkdir -p $MODEL_DIR/average
python3 $FAIRSEQ/scripts/average_checkpoints.py --inputs $MODEL_DIR --output $MODEL_DIR/average/average.pt --num-update-checkpoints 8


######################################
# Generate
######################################
# decode
B=`basename $TEST_SRC`

python3 $FAIRSEQ/generate.py $DATA_DIR \
    --gen-subset test \
    --path $MODEL_DIR/average/average.pt \
    --max-tokens 1000 \
    --beam 6 \
    --lenpen 1.0 \
    --log-format simple \
    --remove-bpe \
    | tee $MODEL_DIR/$B.hyp

grep "^H" $MODEL_DIR/$B.hyp | sed 's/^H-//g' | sort -n | cut -f3 > $MODEL_DIR/$B.true
cat $MODEL_DIR/$B.true | spm_decode --model=$SPM_MODEL --input_format=piece > $MODEL_DIR/$B.true.detok


######################################
# Evaluation
######################################
cat $MODEL_DIR/$B.true.detok | sacrebleu --tokenize=intl $TEST_TRG_RAW | tee -a $MODEL_DIR/test.log
