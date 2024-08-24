export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sdsd_in.yaml" \
  --log_dir="./log/train/sdsd_in/" \
  --alsologtostderr=True 
