import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
#from tensorflow.contrib.layers import flatten

def defineFNN(next_feat_pl,NUM_CLASSES):
	with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/257)),reuse=tf.AUTO_REUSE),tf.variable_scope('FNN'):
		x = slim.fully_connected(next_feat_pl, num_outputs=2048,scope='fc1')
		x = slim.fully_connected(x, num_outputs=2048,scope='fc2')
		x = slim.fully_connected(x, num_outputs=2048,scope='fc3')

		preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc_out')
		#preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc5')

	return tf.identity(preds,name='preds2')
