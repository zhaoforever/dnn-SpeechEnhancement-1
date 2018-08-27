import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def defineFFN(next_feat_pl,NUM_CLASSES,nLayer,keepProb,conigN):
	with slim.arg_scope([slim.fully_connected],activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/257)),reuse=tf.AUTO_REUSE),tf.variable_scope('FFN'):

		for i in range(1,1+nLayer):
			next_feat_pl = slim.fully_connected(next_feat_pl,num_outputs=1024,scope=('fc'+str(i)+'_'+str(conigN)))
			next_feat_pl = tf.nn.dropout(next_feat_pl,keepProb)

		preds = slim.fully_connected(next_feat_pl,activation_fn=None,num_outputs=NUM_CLASSES,scope=('fcOut'+str(conigN)))

	return tf.identity(preds,name='preds2')
