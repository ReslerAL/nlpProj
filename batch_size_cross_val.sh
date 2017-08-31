#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
for b in 25 50 100 200
do
  echo "Train with batch_size $b"
  python3 code/train.py -batch_size $b
done

