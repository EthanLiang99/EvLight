#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sdsd_in_vid.yaml" \
  --log_dir="./log/train/sdsd_in_vid/" \
  --alsologtostderr=True
