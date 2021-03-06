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
from scipy.io import wavfile
### Our functions ###
import FFN_Model_Cond_Dropout
import featureProcessing
import dataStatistics
import modelParameters as mp
import utils

savedModelPath = "./bestHyperparameterModel/run16/"

for root, dirs, files in os.walk(savedModelPath + 'predictionFiles/', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

tf.reset_default_graph()

trainingStats = np.load(savedModelPath + "trainingStatistics.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

dataPath = "C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/"

feat_root_test = dataPath + 'testInput/'
label_root_test  = dataPath + 'testReference/'

allFilesTest = os.listdir(feat_root_test)
#random.shuffle(allFilesTest)
#allFilesTest = allFilesTest[:10]

### UNFREEZE MODEL ###
frozen_graph= savedModelPath + "myFrozenModel.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def,input_map=None,return_elements=None,	name="")

preds = graph.get_tensor_by_name("out/BiasAdd:0")
next_feat_pl = graph.get_tensor_by_name("next_feat_pl:0")

n = 1
with tf.Session(graph=graph) as sess:

    for file in allFilesTest:

        finalPreds = np.empty((0,mp.NUM_CLASSES))

        filePathFeat_test = feat_root_test + file
        filePathRef_test = label_root_test + file

        features_test,features_phi_test = featureProcessing.featureExtraction(filePathFeat_test,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUM_CLASSES,featMean,featStd)

        idx1test = 0

        for idx in range(0,features_test.shape[0]):

            ### Inference ###
            next_feat = np.reshape(features_test[idx,:],[-1,features_test.shape[1]])

            fetches_test = [preds]
            feed_dict_test = {next_feat_pl: next_feat}

            res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)

            finalPreds = np.concatenate((finalPreds,res_test[0]),axis=0)

            finalPreds0 = finalPreds.T

        y = featureProcessing.features2samples(finalPreds0,features_phi_test,featMean,featStd,12000,mp.NFFT,mp.STFT_OVERLAP)

        y =  y*32767
        y = y.astype(np.int16)
        newWavName = file.replace('feat','pred')
        wavfile.write(savedModelPath + 'predictionFiles/' + newWavName,12000,y)
        print('File ' + str(n) + ' done!')
        n += 1

print('Inference done!')

import qualityAssessment

#sd.play(y,12000)
