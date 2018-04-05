import os
import sys

import numpy as np
import random
import cPickle
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class LeNet:
    def __init__(self, x, num_features=10):
        self.num_features = num_features
        self.x = x
        self.build()

    def build(self):
        # Hyperparameters
        mu = 0
        sigma = 0.1
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma), name='conv1_w')
        self.conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
        self.conv1 = tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_b
        # Activation.
        self.relu1 = tf.nn.relu(self.conv1)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        self.pool_1 = tf.nn.max_pool(self.relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma), name='conv2_w')
        self.conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
        self.conv2 = tf.nn.conv2d(self.pool_1, self.conv2_w, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_b
        # Activation.
        self.relu2 = tf.nn.relu(self.conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        self.pool_2 = tf.nn.max_pool(self.relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Flatten. Input = 5x5x16. Output = 400.
        self.fla = flatten(self.pool_2)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name='fc1_w')
        self.fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
        self.fc1 = tf.matmul(self.fla, self.fc1_w) + self.fc1_b
        # Activation.
        self.relu3 = tf.nn.relu(self.fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name='fc2_w')
        self.fc2_b = tf.Variable(tf.zeros(84), name='fc2_b')
        self.fc2 = tf.matmul(self.relu3, self.fc2_w) + self.fc2_b
        # Activation.
        self.relu4 = tf.nn.relu(self.fc2)
        # Layer 5: Fully Connected. Input = 84. Output = number of features.
        self.fc3_w = tf.Variable(tf.truncated_normal(shape=(84, self.num_features), mean=mu, stddev=sigma), name='fc3_w')
        self.fc3_b = tf.Variable(tf.zeros(self.num_features), name='fc3_b')
        self.y = tf.matmul(self.relu4, self.fc3_w) + self.fc3_b
        self.out = self.y
        self.probs = tf.nn.softmax(self.y, name='probs')
