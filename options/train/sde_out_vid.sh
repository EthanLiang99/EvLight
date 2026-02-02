#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sde_out_vid.yaml" \
  --log_dir="./log/train/sde_out_vid/" \
  --alsologtostderr=True
