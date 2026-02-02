#!/bin/bash

export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/test/sde_in_vid.yaml" \
  --log_dir="./log/test/sde_in_vid/" \
  --alsologtostderr=True \
  --VISUALIZE=False 
