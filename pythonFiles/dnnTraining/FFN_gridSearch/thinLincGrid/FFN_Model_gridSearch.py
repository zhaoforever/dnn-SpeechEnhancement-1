import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def defineFFN(next_feat_pl,NUM_CLASSES,slimStack,conigN):
	with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/257)),reuse=tf.AUTO_REUSE),tf.variable_scope('FFN'):

		x = slim.stack(next_feat_pl, slim.fully_connected,slimStack,scope=('fc'+str(conigN)))

		preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope=('fcOut'+str(conigN)))

	return tf.identity(preds,name='preds2')
