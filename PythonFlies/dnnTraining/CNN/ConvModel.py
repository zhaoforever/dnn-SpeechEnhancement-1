import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
#from tensorflow.contrib.layers import flatten

def defineConv(next_feat_pl,NUM_CLASSES,dropProb):
	with slim.arg_scope([slim.fully_connected,slim.conv2d],reuse=tf.AUTO_REUSE,activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/1024))),tf.variable_scope('CNN'):
		conv1 = slim.conv2d(next_feat_pl, 32, [5,5],scope='conv1')

		pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
  
		conv2 = slim.conv2d(pool1, 64, [5,5],scope='conv2')

		pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
  
		x = slim.flatten(pool2)
		x = slim.fully_connected(x, num_outputs=2048,scope='fc1')
  
		x = tf.nn.dropout(x,dropProb)
  
		x = slim.fully_connected(x, num_outputs=2048,scope='fc2')
  
		x = tf.nn.dropout(x,dropProb)
  
		x = slim.fully_connected(x, num_outputs=2048,scope='fc3')
  
		x = tf.nn.dropout(x,dropProb)  
  
		preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc_out')
		#preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc5')

	return tf.identity(preds,name='preds1')
