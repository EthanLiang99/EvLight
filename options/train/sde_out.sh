export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/train/sde_out.yaml" \
  --log_dir="./log/train/sde_out/" \
  --alsologtostderr=True 
