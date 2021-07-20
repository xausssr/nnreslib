from . import BACKEND, Backends

if BACKEND == Backends.TF:
    from .tf.loss_functions import hinge, logloss, mse, sigmoid_cce, softmax_cce
else:
    raise ImportError(f"Unsupported backend: {BACKEND}")

__all__ = ["mse", "sigmoid_cce", "softmax_cce", "logloss", "hinge"]
