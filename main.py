import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_data, test_data = tf.keras.datasets.mnist.load_data()
X_tr, y_tr = train_data
X_test, y_test = test_data
