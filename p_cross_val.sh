#!/bin/sh
for p in 0.9995 0.9999 0.99995 0.99999 0.999999
do
  echo "Train with p = $p"
  CUDA_VISIBLE_DEVICES=$1 python3 code/train.py -p $p
done
