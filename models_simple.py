import os
import numpy as np
import tensorflow as tf


class Simple:
    def __init__(self, x):

        # xavier initialization for the layers
        init = tf.contrib.layers.xavier_initializer()

        # flattent the input into a vector
        self.x = x
        self.input_flat = tf.reshape(self.x, [-1, 32 * 32 * 1])
        # define fully-connected layers
        self.h_1 = tf.layers.dense(inputs=self.input_flat, units=1000,
                              activation=tf.nn.relu, kernel_initializer=init)
        self.h_2 = tf.layers.dense(inputs=self.h_1, units=1000,
                              activation=tf.nn.relu, kernel_initializer=init)
        self.y = tf.layers.dense(inputs=self.h_2, units=10, kernel_initializer=init)
        # add softmax layer for classification
        self.out = self.y
        self.probs = tf.nn.softmax(self.y, name='probs')
