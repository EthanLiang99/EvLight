export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/release/sdsd_out.yaml" \
  --log_dir="./log/release/sdsd_out/" \
  --alsologtostderr=True \
  --VISUALIZE=False
