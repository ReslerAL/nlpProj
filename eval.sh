#!/bin/sh
# run example "./eval.sh 1503569471412 0.5 false"
export CUDA_VISIBLE_DEVICES=0
WORD_EMBDS_FILE=paragram-phrase-XXL.txt
EVAL_DATA_FILE=eval_data_nlmlproj.tsv
MODEL_DIR=./saved/$1/
echo "Simple model evaluation:\n"
python3 code/evaluate.py -model simple -embdsfile $WORD_EMBDS_FILE -evaldata $EVAL_DATA_FILE -elimination $2 -softmax $3
echo "\nLstm model ($1) evaluation:\n"
python3 code/evaluate.py -model lstm -dir $MODEL_DIR -evaldata $EVAL_DATA_FILE -elimination $2 -softmax $3