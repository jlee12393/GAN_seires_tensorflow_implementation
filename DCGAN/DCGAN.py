import tensorflow as tf

W= tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())