#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sde_in_vid.yaml" \
  --log_dir="./log/train/sde_in_vid/" \
  --alsologtostderr=True
