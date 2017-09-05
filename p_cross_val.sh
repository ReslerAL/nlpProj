#!/bin/sh
for p in 0.9995 0.9999 1.0
do
  echo "Train with p = $p"
  CUDA_VISIBLE_DEVICES=$1 python3 code/train.py -p $p
done
