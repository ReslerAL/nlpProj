#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
python3 code/train.py $@