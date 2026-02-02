import json
import os

import yaml
from absl import app, flags, logging
from absl.logging import info
from easydict import EasyDict
from pudb import set_trace

from egllie.core.launch import ParallelLaunch
from egllie.core.launch_vid import ParallelLaunchVid

FLAGS = flags.FLAGS

flags.DEFINE_string("yaml_file", None, "The config file.")
flags.DEFINE_string("RESUME_PATH", None, "The RESUME.PATH.")
flags.DEFINE_string("RESUME_TYPE", None, "The RESUME.PATH.")
flags.DEFINE_boolean("RESUME_SET_EPOCH", False, "The RESUME.PATH.")
flags.DEFINE_boolean("TEST_ONLY", False, "The test only.")
flags.DEFINE_boolean("PUDB", False, "The debug switch.")
flags.DEFINE_boolean(f"VISUALIZE", False, "The visualization switch.")
#
flags.DEFINE_integer(f"TRAIN_BATCH_SIZE", None, "The train batch size.")
flags.DEFINE_integer(f"VAL_BATCH_SIZE", None, "The test batch size.")


def init_config(yaml_path):
    """
    This is the config file for the project.
    :paparam yaml_path: The path of the yaml file.
    :return: The config with Easy Dict.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    # 0. logging
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    config["SAVE_DIR"] = FLAGS.log_dir
    # 1. Resume
    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
        config["RESUME"]["SET_EPOCH"] = FLAGS.RESUME_SET_EPOCH
    # 3. VISUALIZATION
    config["VISUALIZE"] = FLAGS.VISUALIZE
    # 4. Update batch size
    if FLAGS.TRAIN_BATCH_SIZE:
        info(f"Update TRAIN_BATCH_SIZE to {FLAGS.TRAIN_BATCH_SIZE}")
        config["TRAIN_BATCH_SIZE"] = FLAGS.TRAIN_BATCH_SIZE
    if FLAGS.VAL_BATCH_SIZE:
        info(f"Update VAL_BATCH_SIZE to {FLAGS.VAL_BATCH_SIZE}")
        config["VAL_BATCH_SIZE"] = FLAGS.VAL_BATCH_SIZE
    # 5. TEST_ONLY
    if FLAGS.TEST_ONLY:
        config["TEST_ONLY"] = FLAGS.TEST_ONLY
    # 6. Debug
    if FLAGS.PUDB:
        set_trace()

    info(f"Launch Config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)


def main(args):
    # if using pudb to debug the code, please uncomment the following line.
    # set_trace()
    config = init_config(FLAGS.yaml_file)
    # 0. logging
    # 1. init launcher and run
    if config.get('IS_VIDEO', False):
        launcher = ParallelLaunchVid(config)
    else:
        launcher = ParallelLaunch(config)
    launcher.run()


if __name__ == "__main__":
    app.run(main)
