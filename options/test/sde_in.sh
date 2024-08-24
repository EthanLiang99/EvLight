export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/release/sde_in.yaml" \
  --log_dir="./log/release/sde_in/" \
  --alsologtostderr=True  \
