import logging
import os

_logger = logging.getLogger(__name__)

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa isort:skip
import tensorflow as tf  # noqa isort:skip pylint:disable=wrong-import-position

tf.compat.v1.disable_eager_execution()

_logger.debug("%s loaded", __name__)

__version__ = tf.__version__
