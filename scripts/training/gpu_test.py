import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
