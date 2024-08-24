export PYTHONPATH="./":$PYTHONPATH

python egllie/main.py \
  --yaml_file="options/release/sdsd_in.yaml" \
  --log_dir="./log/release/sdsd_in/" \
  --alsologtostderr=True \
  --VISUALIZE=True
