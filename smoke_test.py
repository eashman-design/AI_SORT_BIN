import cv2
import tensorflow as tf

print("OpenCV:", cv2.__version__)
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Simple GPU op
a = tf.random.normal([1024, 1024])
b = tf.random.normal([1024, 1024])
c = tf.matmul(a, b)
print("MatMul device:", c.device)
print("OK:", float(tf.reduce_sum(c).numpy()))
