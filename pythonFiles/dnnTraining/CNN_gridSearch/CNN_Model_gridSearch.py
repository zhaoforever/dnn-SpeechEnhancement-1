import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
#from tensorflow.contrib.layers import flatten

def convLayer2d(x,filterSize,convName,poolName):
	conv = slim.conv2d(x, filterSize, [5,5], scope=convName)
	pool = slim.max_pool2d(conv, [2, 2], scope=poolName)
	return (pool)


def defineConv(next_feat_pl,NUM_CLASSES,convLayers,conigN,keepProb):
	with slim.arg_scope([slim.fully_connected,slim.conv2d],reuse=tf.AUTO_REUSE,activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/1024))),tf.variable_scope('CNN'):

		for i in range(convLayers):
			convName = 'conv_%d_' %(i+1)+str(conigN)
			poolName = 'pool_%d_' %(i+1)+str(conigN)
			next_feat_pl = convLayer2d(next_feat_pl,32,convName,poolName)

		x = slim.flatten(next_feat_pl)
		x = slim.fully_connected(x, num_outputs=1024,scope='fc1_%d_' %convLayers+str(conigN))
		x = tf.nn.dropout(x,keepProb)
  
		x = slim.fully_connected(x, num_outputs=1024,scope='fc2_%d_' %convLayers+str(conigN))
		x = tf.nn.dropout(x,keepProb)

		preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,activation_fn=None,scope='fcOut_%d_' %convLayers+str(conigN))

	return tf.identity(preds,name='preds2')
