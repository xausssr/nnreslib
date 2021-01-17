from . import BACKEND, Backends

if BACKEND == Backends.TF:
    from .tf.graph import (
        Dataset,
        GradientDescentOptimizer,
        Operation,
        PlaceholderType,
        Session,
        Tensor,
        TensorArray,
        Variable,
        VariableType,
        assign,
        concat,
        conv2d,
        eye,
        global_variables_initializer,
        gradients,
        graph_function,
        hessians,
        inv,
        matmul,
        max_pool,
        multiply,
        placeholder,
        reduce_mean,
        reshape,
        split,
        square,
        squeeze,
        zeros,
    )
else:
    raise ImportError(f"Unsupported backend: {BACKEND}")

__all__ = [
    "Dataset",
    "GradientDescentOptimizer",
    "Operation",
    "PlaceholderType",
    "Session",
    "Tensor",
    "TensorArray",
    "Variable",
    "VariableType",
    "assign",
    "concat",
    "conv2d",
    "eye",
    "global_variables_initializer",
    "gradients",
    "graph_function",
    "hessians",
    "inv",
    "matmul",
    "max_pool",
    "multiply",
    "placeholder",
    "reduce_mean",
    "reshape",
    "split",
    "square",
    "squeeze",
    "zeros",
]
