import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def defineFFN(next_feat_pl,NUM_CLASSES,keepProb):
	# with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/257))):
	# #,reuse=tf.AUTO_REUSE
	# #with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, 	tf.variable_scope('FFN')):
	#
	# 	x = slim.fully_connected(next_feat_pl, num_outputs=2048)
	# 	x = tf.nn.dropout(x,keepProb)
	#
	# 	x = slim.fully_connected(x, num_outputs=2048)
	# 	x = tf.nn.dropout(x,keepProb)
	#
	# 	out = slim.fully_connected(x, activation_fn=None,num_outputs=NUM_CLASSES)
	#
	# 	preds1 = tf.nn.leaky_relu(out,alpha=0.2,name='preds1')
	#
	# 	#preds = tf.identity(out,name='preds')
	# return preds1
	# 
	x1 = tf.layers.dense(next_feat_pl,2048,activation=tf.nn.leaky_relu,name='fc1')
	x1 = tf.nn.dropout(x1,keepProb)

	x2 = tf.layers.dense(x1,2048,activation=tf.nn.leaky_relu,name='fc2')
	x2 = tf.nn.dropout(x2,keepProb)

	out = tf.layers.dense(x2,NUM_CLASSES,activation=None,name='out')

	return out
