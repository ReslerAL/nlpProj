#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
EVAL_DATA_FILE=eval_data_nlmlproj.tsv
WORD_EMBDS_FILE=paragram-phrase-XXL.txt
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	echo "Evaluate for p = $i"
	python3 code/evaluate.py -model simple -embdsfile $WORD_EMBDS_FILE -evaldata $EVAL_DATA_FILE -elimination $i -softmax $1		
done
