#!/bin/sh
models=$(ls saved/) 
for model in $models
do
	echo "evaluting model $model...\n" 
	python3 code/evaluate.py -model lstm -dir ./saved/${model}/ -evaldata evaluation_data.tsv -elimination $1 -verbose false
done
