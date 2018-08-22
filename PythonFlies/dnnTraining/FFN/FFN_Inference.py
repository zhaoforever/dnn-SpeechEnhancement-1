import sys
sys.path.append("C:/Users/Mikkel/Desktop/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
### Our functions ###
import FFN_Model_Cond_Dropout
import FeatureExtraction
import dataStatistics
import modelParameters as mp

tf.reset_default_graph()

dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
trainingStats = np.load(dataPath + "trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

filePath_input = "C:/Users/s123028/dataset8_MulitTfNoise/TIMIT_val_feat/748_val_feat.wav"
filePath_target = "C:/Users/s123028/dataset8_MulitTfNoise/TIMIT_val_ref/748_val_ref.wav"

### Feature Extraction ###
features,features_phi = FeatureExtraction.FeatureExtraction(filePath_input,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

labels,labels_phi = FeatureExtraction.FeatureExtraction(filePath_target,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

### UNFREEZE MODEL ###
frozen_graph="./savedModelsWav/myFrozenModel.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def,input_map=None,return_elements=None,	name="")

preds = graph.get_tensor_by_name("out/BiasAdd:0")
next_feat_pl = graph.get_tensor_by_name("next_feat_pl:0")

finalPreds = np.empty((0,mp.NUM_CLASSES))

with tf.Session(graph=graph) as sess:

	for idx in range(0,features.shape[0]):

		### Inference ###
		next_feat = np.reshape(features[idx,:],[-1,features.shape[1]])

		fetches_test = [preds]
		feed_dict_test = {next_feat_pl: next_feat}

		res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

		finalPreds = np.concatenate((finalPreds,res_test[0]),axis=0)

	finalPreds0 = finalPreds.T
print('Training done!')




plt.pcolormesh(features.T)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.pcolormesh(finalPreds0)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.pcolormesh(labels.T)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
