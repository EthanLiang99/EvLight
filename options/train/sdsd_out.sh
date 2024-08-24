export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sdsd_out.yaml" \
  --log_dir="./log/train/sdsd_out/" \
  --alsologtostderr=True 
