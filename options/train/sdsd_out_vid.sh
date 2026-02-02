#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sdsd_out_vid.yaml" \
  --log_dir="./log/train/sdsd_out_vid/" \
  --alsologtostderr=True
