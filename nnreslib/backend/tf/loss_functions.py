import tensorflow as tf

mse = tf.compat.v1.losses.mean_squared_error
# mse = lambda y_pred, y_true: tf.reduce_mean(tf.square(y_true - y_pred))
sigmoid_cce = tf.compat.v1.losses.sigmoid_cross_entropy
softmax_cce = tf.compat.v1.losses.softmax_cross_entropy
logloss = tf.compat.v1.losses.log_loss
hinge = tf.compat.v1.losses.hinge_loss

__all__ = ["mse", "softmax_cce", "logloss", "hinge"]
