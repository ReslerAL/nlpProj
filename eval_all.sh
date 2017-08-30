#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
models=$(ls saved/) 
for model in $models
do
	echo "evaluting model $model" 
	python3 code/evaluate.py -model lstm -dir ./saved/${model}/ -evaldata eval_data_nlmlproj.tsv -elimination $1 -softmax $2
	echo '\n'
done
