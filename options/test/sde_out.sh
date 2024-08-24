export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/release/sde_out.yaml" \
  --log_dir="./log/release/sde_out/" \
  --alsologtostderr=True \
  --VISUALIZE=True
