import utils
import pesq_calc
import MCC_calc
import os, sys

import numpy as np
import matplotlib.pyplot as plt


dataPath_feat_root = 'C:/Users/TobiasToft/Desktop/AudioFilesTest/TIMIT_feat'
dataPath_ref_root = 'C:/Users/TobiasToft/Desktop/AudioFilesTest/TIMIT_ref'

### Dataset and feature extraction parameters ###
NFFT = 512
DATASET_SIZE_VAL = 3

# initialize empty arrays
MCC_all = []
segMCC_all = np.empty((int(NFFT/2+1),0), int)
pesqScore_all = []


allFiles = os.listdir(dataPath_feat_root)
allFiles = allFiles[0:DATASET_SIZE_VAL]

for file in allFiles:
	filePathFeat = dataPath_feat_root + '/' + file
	filePathRef = dataPath_ref_root + '/' + file

	MCC,segMCC,freq = MCC_calc.MCC_calc(filePathRef.replace('feat','ref'),filePathFeat,NFFT)
	MCC_all = np.append(MCC_all,MCC)
	segMCC_all = np.append(segMCC_all,segMCC,axis=1)

	pesqScore = pesq_calc.pesq_calc(filePathRef.replace('feat','ref'),filePathFeat)
	pesqScore_all = np.append(pesqScore_all,pesqScore)



MCC_mean = np.mean(MCC_all)
segMCC_mean = np.mean(segMCC_all,axis=1)
pesqScore_mean = np.mean(pesqScore_all)


print("Size of dataset: " + str(np.size(allFiles)) + " wav files" + "\n")
print("MCC Average: " + str(MCC_mean))
print("PESQ Average: " + str(pesqScore_mean))




### Figures ###

plt.semilogx(freq,segMCC_mean, linewidth=2,color='b')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude Correlation")
plt.title("MCC")
plt.grid(linestyle='--')
plt.show()
