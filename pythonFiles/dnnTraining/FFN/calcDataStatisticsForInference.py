import sys, os
import numpy as np
import time
#sys.path.append("/home/paperspace/Desktop/pythonFiles/dataProcessing/")
sys.path.append("C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dataProcessing/")
import dataStatistics
import modelParameters as mp
dataPath = "C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/"
#dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
label_root_train = dataPath + "trainingReference/"

allFilesTrain = os.listdir(label_root_train)
numFilesTrain = len(allFilesTrain)


tic = time.time()
featMean, featStd = dataStatistics.calcMeanAndStd(label_root_train,len(allFilesTrain),mp.NFFT,mp.STFT_OVERLAP,mp.BIN_WIZE)
np.save('./bestHyperparameterModel/run16/trainingStatistics.npy',[featMean,featStd])
toc = time.time()
print(toc-tic)
