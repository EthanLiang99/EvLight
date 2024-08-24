export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sde_in.yaml" \
  --log_dir="./log/train/sde_in/" \
  --alsologtostderr=True 
