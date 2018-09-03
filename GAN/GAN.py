import tensorflow as tf
import numpy as np

#loss function modification
# log(1-D(z)) is close to 0 at first which hinders learning process
#So, instead of minimizing log(1-D(z)), we mzximize log(D(z))

datadirectory="C:/Users/jlee1/Desktop/Dataset/img_align_celeba"

W= tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())