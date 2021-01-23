from . import BACKEND, Backends

if BACKEND == Backends.TF:
    from .tf.activation_functions import relu, sigmoid, softmax, tanh
else:
    raise ImportError(f"Unsupported backend: {BACKEND}")

__all__ = ["relu", "sigmoid", "softmax", "tanh"]
