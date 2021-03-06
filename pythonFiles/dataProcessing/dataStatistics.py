import numpy as np
import featureExtraction
import os
import utils


#dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/TIMIT_train_feat1/"
#NFFT = 512
#STFT_OVERLAP = 0.75
#BIN_WIZE = True

def calcMeanAndStd(dataPath,NUM_FILES,NFFT,STFT_OVERLAP,BIN_WIZE):
	allFiles = os.listdir(dataPath)
	allFiles = allFiles[:NUM_FILES]
	#averageOld = 0
	#stdOld = 0
	count = 0

	featMean = 0
	featStd = 0

	for file in allFiles:
		count += 1

		filePath = dataPath + file

		x, fs = utils.wavToSamples(filePath)
		x = utils.adjustSNR(x,60)

		features,feature_phi,_,_ = utils.STFT(x,fs,NFFT,int(NFFT*STFT_OVERLAP))
		features = np.float32(features)
		features = np.log10(features + 1e-7)

		if BIN_WIZE:
			featMean = np.asarray(np.mean(features,1))
			featStd = np.asarray(np.std(features,1))
		else:
			featMean = featMean + np.mean(features)
			featStd =  featStd 	+ np.std(features)

		#print((featMean + averageOld)/count)
		#averageOld = averageOld + (featMean + averageOld)/count
		#print(averageOld)
		#stdOld = stdOld + (featStd + stdOld)/count
	featMean = featMean/count
	featStd = featStd/count

	#averageNew = averageOld
	#stdNew = stdOld

	return featMean, featStd
