#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6
do
  echo "Train with margin $i"
  python3 code/train.py -margin $i
done
