#!/bin/sh
for i in 0.00005 0.0001 0.0005 0.001 0.005 0.01
do
	echo "Train for learning rate = $i"
	CUDA_VISIBLE_DEVICES=$1 python3 code/train.py -lr $i
done

