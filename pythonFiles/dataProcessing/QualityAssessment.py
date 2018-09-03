import os, sys
sys.path.append("C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dataProcessing/")
import utils
import pesq_calc
import MCC_calc
import time
import random

import numpy as np
import matplotlib.pyplot as plt

predictionPath = "C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dnnTraining/FFN/bestHyperparameterModel/run16/predictionFiles/"
inputPath = 'C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/testInput/'
referencePath = 'C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/testReference/'

### Dataset and feature extraction parameters ###
NFFT = 512

# initialize empty arrays
MCC_all_pred = []
MCC_all_input = []
segMCC_all_pred = np.empty((int(NFFT/2+1),0), int)
segMCC_all_input = np.empty((int(NFFT/2+1),0), int)
pesqScore_all_pred = []
pesqScore_all_input = []



allFiles = os.listdir(predictionPath)
random.shuffle(allFiles)
n = 1
for file in allFiles:
	tic = time.time()
	filePathPrediction = predictionPath + file
	filePathInput = inputPath + file.replace('pred','feat')
	filePathReference = referencePath + file.replace('pred','ref')

	### MCC ###

	# Prediction
	MCC_pred,segMCC_pred,freq = MCC_calc.MCC_calc(filePathReference,filePathPrediction,NFFT)
	MCC_all_pred.append(MCC_pred)
	segMCC_all_pred = np.append(segMCC_all_pred,segMCC_pred,axis=1)
	# Input
	MCC_input,segMCC_input,freq = MCC_calc.MCC_calc(filePathReference,filePathInput,NFFT)
	MCC_all_input.append(MCC_input)
	segMCC_all_input = np.append(segMCC_all_input,segMCC_input,axis=1)



	pesqScore_pred = pesq_calc.pesq_calc(filePathReference,filePathPrediction)
	pesqScore_all_pred = np.append(pesqScore_all_pred,pesqScore_pred)

	pesqScore_input = pesq_calc.pesq_calc(filePathReference,filePathInput)
	pesqScore_all_input = np.append(pesqScore_all_input,pesqScore_input)

	print(n)
	n+=1
	toc = time.time()
	print(toc-tic)

MCC_mean_pred = np.mean(MCC_all_pred)
MCC_mean_input = np.mean(MCC_all_input)
segMCC_mean_pred = np.mean(segMCC_all_pred,axis=1)
segMCC_mean_input = np.mean(segMCC_all_input,axis=1)

pesqScore_mean_pred = np.mean(pesqScore_all_pred)
pesqScore_mean_input = np.mean(pesqScore_all_input)


print("Size of dataset: " + str(np.size(allFiles)) + " wav files" + "\n")
print("            MCC   PESQ")
print("Input:      " + str(np.round(MCC_mean_input,2)) + " " + str(np.round(pesqScore_mean_input,2)))
print("Prediction: " + str(np.round(MCC_mean_pred,2)) + " " + str(np.round(pesqScore_mean_pred,2)))
print("Difference: " + str(np.round((MCC_mean_pred-MCC_mean_input),2)) + " " + str(np.round((pesqScore_mean_pred-pesqScore_mean_input),2)))





### Figures ###

plt.semilogx(freq,segMCC_mean_input,freq,segMCC_mean_pred,linewidth=1)
plt.legend(['input','pred'])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude Correlation")
plt.title("MCC")
plt.grid(linestyle='--')
#plt.xlim(50, 8000)
plt.show()
