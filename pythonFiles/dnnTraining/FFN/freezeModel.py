import tensorflow as tf
import numpy as np
tf.reset_default_graph()

savedModelPath = './savedModelsTrain/'

saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(savedModelPath) + '.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint(savedModelPath))

#tf.global_variables()

#tf.get_variable('variance_scaling')
#for node in input_graph_def.node:
	#print(node.name)

#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


#latest_ckp = tf.train.latest_checkpoint('./savedModelsWav/')
#print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='',all_tensor_names=True)


# Output from the model that needs to be freezed
output_node_names = "out/BiasAdd"
output_graph_def = tf.graph_util.convert_variables_to_constants(sess,input_graph_def,output_node_names.split(","))
output_graph=savedModelPath + "myFrozenModel.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
