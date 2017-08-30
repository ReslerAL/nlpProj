#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
models=$(ls saved/) 
for model in $models
do
	echo "evaluting model $model...\n" 
	python3 code/evaluate.py -model lstm -dir ./saved/${model}/ -evaldata eval_data_nlmlproj.tsv -elimination $1 -verbose false
done
