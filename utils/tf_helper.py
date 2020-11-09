import os

# Supress TensorFlow debug output
# If it will be in docker, need put to Dockerfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # noqa isort:skip

import tensorflow as tf  # noqa isort:skip
from tensorflow.python.util import deprecation  # noqa isort:skip


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
