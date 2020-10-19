import tensorflow as tf

if tf.version.VERSION < "2.3":
    raise ImportError("Requires tensorflow 2.3 or later")
