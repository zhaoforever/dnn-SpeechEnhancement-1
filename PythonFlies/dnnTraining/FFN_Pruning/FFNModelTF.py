import tensorflow as tf
import numpy as np


def defineFFN(next_feat_pl,NUM_UNITS,NUM_CLASSES,keepProb):

	x1 = tf.layers.masked_fully_connected(next_feat_pl,NUM_UNITS,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.leaky_relu,name='fc1')
	x1 = tf.nn.dropout(x1,keepProb)

	x2 = tf.layers.masked_fully_connected(x1,NUM_UNITS,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.leaky_relu,name='fc2')
	x2 = tf.nn.dropout(x2,keepProb)

	out = tf.layers.masked_fully_connected(x2,NUM_CLASSES,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=None,name='out')

	return out
