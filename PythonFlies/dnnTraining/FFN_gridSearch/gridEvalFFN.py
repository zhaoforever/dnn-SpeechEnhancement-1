import tensorflow as tf
import numpy as np
import FFN_Model_gridSearch
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import random
import imp

sys.path.append("C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/")
import utils
utils = imp.reload(utils)

slim = tf.contrib.slim


dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
data_root = dataPath + 'TIMIT_val_feat'
ref_root =  dataPath + 'TIMIT_val_ref'


NUM_CLASSES = 257
features_placeholder = tf.placeholder(np.float32, (None,NUM_CLASSES))

dataFeatures = tf.data.Dataset.from_tensor_slices(features_placeholder)

testDataset = tf.data.Dataset.zip(dataFeatures)

testDataset = testDataset.batch(1)
testIterator = testDataset.make_initializable_iterator()
next_testFeat = testIterator.get_next()

next_testElement = next_testFeat

next_feat_pl = tf.placeholder(next_testFeat.dtype,next_testFeat.shape)
keepProb = tf.placeholder(tf.float32)

nUnits = 2048
layers = 1

configN = layers


epoch_num = 1
allFiles = os.listdir(data_root)
#random.shuffle(allFiles)
with tf.Session() as sess:
	preds = FFN_Model_gridSearch.defineFFN(next_feat_pl,NUM_CLASSES,layers,keepProb,configN)
	sess.run(tf.global_variables_initializer())
	model_variables = slim.get_model_variables()
	new_saver = tf.train.Saver(model_variables)
	new_saver.restore(sess, tf.train.latest_checkpoint('./rate_e6/'))

	count = 0
	for file in allFiles:
		filePath = data_root + '/' + file
		filePath2 = ref_root + '/' + file

		if count == 5:
			break

		x, fs = utils.wavToSamples(filePath)
		xRef, fs = utils.wavToSamples(filePath2.replace('feat','ref'))
		#x = x[:int(fs*30)]
		#xRef = xRef[:int(fs*30)]

		featuresOrg, X_phi =  utils.featureExtractMag512(x,fs,dataPath)
		features = featuresOrg
		labels, L_phi =  utils.featureExtractMag512(xRef,fs,dataPath)

		finalPreds = []

		for epoch in range(0,epoch_num):
			sess.run(testIterator.initializer,
	        feed_dict={features_placeholder: features})
			while True:
				try:

					next_feat = sess.run(next_testElement)

					fetches_test = [preds]
					feed_dict_test = {next_feat_pl: next_feat,keepProb: 1.0}

					# running the traning optimizer
					res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)


					finalPreds.append(res_test[0])

					#idx1 +=1
					#idx2 +=1
				except tf.errors.OutOfRangeError:

					break
		finalPreds = np.transpose(np.asarray(finalPreds)[:,-1,:])

		alpha = 1.3
		finalPreds = alpha*finalPreds

		y =  utils.featureReconstructMag512(finalPreds,X_phi,fs,dataPath)
		x2 =  utils.featureReconstructMag512(featuresOrg.T,X_phi,fs,dataPath)
		xRef2 =  utils.featureReconstructMag512(labels.T,L_phi,fs,dataPath)

		count += 1
		predPath = ("./wavPredictionFiles/")
		wavfile.write((predPath + str(count) + 'pred' + '.wav'),fs,y)
		wavfile.write((predPath + str(count) + 'feat' + '.wav'),fs,x2)
		wavfile.write((predPath + str(count) + 'ref'  + '.wav'),fs,xRef2)
		print(count)
print('Done!')
