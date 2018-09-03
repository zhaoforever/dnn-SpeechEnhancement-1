import sys
sys.path.append("C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dataProcessing/")
#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as dsp
### Our functions ###
import FFN_Model_Cond_Dropout
import featureExtraction
import dataStatistics
import modelParameters as mp
import utils

tf.reset_default_graph()

savedModelPath = "./savedModels_HP_Tune/"
trainingStats = np.load(savedModelPath + "trainingStatistics.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

filePath_input = "C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/testInput/tobc_01_feat_11.wav"
filePath_target = "C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/testReference/tobc_01_ref_11.wav"

### Feature Extraction ###
print(dsp.check_COLA('hann',mp.NFFT,int(mp.NFFT*mp.STFT_OVERLAP)))
features,features_phi = featureExtraction.featureExtraction(filePath_input,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUM_CLASSES,featMean,featStd)

labels,labels_phi = featureExtraction.featureExtraction(filePath_target,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUM_CLASSES,featMean,featStd)

### UNFREEZE MODEL ###
frozen_graph= savedModelPath + "myFrozenModel.pb"
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



#
# plt.pcolormesh(features.T)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
#
# plt.pcolormesh(finalPreds0)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
#
# plt.pcolormesh(labels.T)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()



if not finalPreds0.shape == features_phi.shape:
    fillFreqs = np.full((features_phi.shape[0] - finalPreds0.shape[0],features_phi.shape[1]),np.min(finalPreds0))
    finalPreds0 = np.concatenate((finalPreds0,fillFreqs),axis=0)

finalPreds1 = (finalPreds0 * featStd)
finalPreds2 = (finalPreds1 + featMean)
finalPreds3 = 10**(finalPreds2)-1e-7
finalPreds3.shape
features_phi.shape



y = utils.ISTFT(finalPreds3,features_phi,12000,mp.NFFT,int(mp.NFFT*mp.STFT_OVERLAP))
y = utils.adjustSNR(y,60)

plt.plot(y)

sd.play(y,12000)
