import tensorflow as tf
import numpy as np
import CNN_Model_gridSearch
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import random
import imp

#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/")
sys.path.append("C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/")

import utils
utils = imp.reload(utils)

slim = tf.contrib.slim

#dataPath = 'C:/Users/TobiasToft/Desktop/multiSyncWithNoise_2/'
dataPath = 'C:/Users/s123028/dataset8_MulitTfNoise/'

data_root = dataPath + 'TIMIT_val_feat'
ref_root =  dataPath + 'TIMIT_val_ref'


frameWidth = 10
nchannels = 1
batchSize = 1


convLayers = 2
conigN = 2

count = 0
for file in os.listdir(data_root):
	filePath = data_root + '/' + file
	filePath2 = ref_root + '/' + file

	if count == 5:
		break

	x, fs = utils.wavToSamples(filePath)
	xRef, fs = utils.wavToSamples(filePath2.replace('feat','ref'))
	#x = x[int(fs*1):int(fs*10)]
	#xRef = xRef[int(fs*1):int(fs*10)]

	featuresOrg, X_phi =  utils.featureExtractMag512(x,fs,dataPath)
	features = featuresOrg
	labels, L_phi =  utils.featureExtractMag512(xRef,fs,dataPath)

	features = np.concatenate((np.repeat(features[:1,:],frameWidth-1,axis=0),features[:,:]),axis=0)

	features = features[:,:150]
	next_feat_pl = tf.placeholder(tf.float32 ,[batchSize, frameWidth, features.shape[1], nchannels])
	keepProb = tf.placeholder(tf.float32)

	NUM_CLASSES = int(labels.shape[1])

	epoch_num = 1

	finalPreds = []

	idx1 = 0
	idx2 = idx1+frameWidth

	with tf.Session() as sess:
		preds = CNN_Model_gridSearch.defineConv(next_feat_pl,NUM_CLASSES,convLayers,conigN,keepProb)
		sess.run(tf.global_variables_initializer())
		model_variables = slim.get_model_variables()
		new_saver = tf.train.Saver(model_variables)
		#new_saver = tf.train.import_meta_graph('./savedModels/my_test_model.ckpt.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./savedModels/'))
		for epoch in range(0,epoch_num):
			while idx2 <= features.shape[0]:
				try:
					next_feat = np.empty((0,features.shape[1]))
					for n in range(0,batchSize):
						next_feat = np.concatenate((next_feat,features[idx1:idx2,:]),axis=0)

						idx1 +=1
						idx2 +=1

					next_feat = np.reshape(next_feat,[-1,frameWidth,features.shape[1],1])

					fetches_test = [preds]
					feed_dict_test = {next_feat_pl: next_feat,keepProb: 1.0}

					# running the traning optimizer
					res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

					finalPreds.append(res_test[0])

				except IndexError:
					break
	finalPreds0 = np.transpose(np.asarray(finalPreds)[:,-1,:])

	y =  utils.featureReconstructMag512(finalPreds0,X_phi,fs,dataPath)
	y = y/np.max(np.abs(y))
	#y = signal.resample(y,len(y)*3)
	y = np.float32(y)
	yWav = np.asarray([y,y],dtype=np.float32).T

	x2 =  utils.featureReconstructMag512(featuresOrg.T,X_phi,fs,dataPath)
	x2 = x2/np.max(np.abs(x2))
	#x2 = signal.resample(x2,len(x2)*3)
	x2 = np.float32(x2)
	x2Wav = np.asarray([x2,x2],dtype=np.float32).T

	xRef2 =  utils.featureReconstructMag512(labels.T,L_phi,fs,dataPath)
	xRef2 = xRef2/np.max(np.abs(xRef2))
	#xRef2 = signal.resample(xRef2,len(xRef2)*3)
	xRef2 = np.float32(xRef2)
	xRef2Wav = np.asarray([xRef2,xRef2],dtype=np.float32).T

	count += 1

	predPath = "./wavPredsForMultiPESQ/"
	wavfile.write((predPath + str(count) + 'pred' + '.wav'),fs,yWav)
	wavfile.write((predPath + str(count) + 'feat' + '.wav'),fs,x2Wav)
	wavfile.write((predPath + str(count) + 'ref' + '.wav'),fs,xRef2Wav)
	print(count)
print('Done!')
