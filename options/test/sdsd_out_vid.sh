#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/test/sdsd_out_vid.yaml" \
  --log_dir="./log/test/sdsd_out_vid/" \
  --alsologtostderr=True
