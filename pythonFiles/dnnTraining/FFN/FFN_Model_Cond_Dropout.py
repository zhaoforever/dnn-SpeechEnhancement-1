import tensorflow as tf
import numpy as np


def defineFFN(next_feat_pl,NUM_UNITS,NUM_CLASSES,keepProb,is_train):

	initializer = tf.contrib.layers.xavier_initializer()

	x1 = tf.layers.dense(next_feat_pl,NUM_UNITS,kernel_initializer=initializer,activation=tf.nn.leaky_relu,name='fc1')
	#x1 = tf.nn.dropout(x1,keepProb)
	x1 = tf.cond(is_train,lambda: tf.nn.dropout(x1,keepProb),lambda: x1)

	x2 = tf.layers.dense(x1,NUM_UNITS,kernel_initializer=initializer,activation=tf.nn.leaky_relu,name='fc2')
	#x2 = tf.nn.dropout(x2,keepProb)
	x2 = tf.cond(is_train,lambda: tf.nn.dropout(x2,keepProb),lambda: x2)

	out = tf.layers.dense(x2,NUM_CLASSES,kernel_initializer=initializer,activation=None,name='out')

	#alpha = tf.Variable(1,dtype=tf.float32,trainable=True,name="variance_scaling")
	#out = tf.scalar_mul(alpha,out)

	return out
