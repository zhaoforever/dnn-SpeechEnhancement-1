import os
import numpy as np
import utils


### Dataset and feature extraction parameters ###
DATASET_SIZE_TRAIN = 5
DATASET_SIZE_VAL = 2
NFFT = 512
STFT_OVERLAP = 0.75
NUM_CLASSES = int(NFFT/2+1)
AUDIO_dB_SPL = 60


### Path to dataset ###
dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
feat_root_train = dataPath + "TIMIT_train_feat1/"

trainingStats = np.load(dataPath + "trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]


allFilesTrain = os.listdir(feat_root_train)
allFilesTrain = allFilesTrain[0:DATASET_SIZE_TRAIN]

for file in allFilesTrain:
	filePathFeat = feat_root_train + '/' + file
    features = FeatureExtraction()
    print(features.shape)





def FeatureExtraction(filePathFeat,AUDIO_dB_SPL,NFFT,STFT_OVERLAP,featMean,featStd):
	x, fs = utils.wavToSamples(filePathFeat)
	x = utils.adjustSNR(x,AUDIO_dB_SPL)

	features,X_phi,_,_ = utils.STFT(x,fs,NFFT,int(NFFT*STFT_OVERLAP))
	features = np.float32(features)
	features = np.log10(features + 1e-7)
	features = features - featMean
	features = features/featStd

	features = np.transpose(features)

return features
