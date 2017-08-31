#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
for lambda_c in 0.01 0.001 0.0001 0.00001
do
    for lambda_w in 1e-03 1e-04 1e-05 1e-06 1e-07 
    do    
	echo "Train with lambda_c $lambda_c and lambda_w $lambda_w"
        python3 code/train.py -lc $lambda_c -lw $lambda_c
    done 
done
