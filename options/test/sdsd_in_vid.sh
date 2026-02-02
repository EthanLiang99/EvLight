#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/test/sdsd_in_vid.yaml" \
  --log_dir="./log/test/sdsd_in_vid/" \
  --alsologtostderr=True
