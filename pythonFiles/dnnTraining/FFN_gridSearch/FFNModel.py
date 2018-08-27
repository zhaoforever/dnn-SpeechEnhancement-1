import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def defineFFN(next_feat_pl,NUM_CLASSES,phase):
	with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/257)),reuse=tf.AUTO_REUSE),tf.variable_scope('FFN'):
	#with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, 	tf.variable_scope('FFN')):

		x = slim.fully_connected(next_feat_pl, num_outputs=1024,scope='fc1')
		#x = tf.contrib.layers.batch_norm(x,center=True, scale=True,is_training=phase,scope='bn1')

		x = slim.fully_connected(x, num_outputs=1024,scope='fc2')
		#x = tf.contrib.layers.batch_norm(x,center=True, scale=True,is_training=phase,scope='bn2')

		x = slim.fully_connected(x, num_outputs=1024,scope='fc3')

		preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc10')

	return tf.identity(preds,name='preds2')
