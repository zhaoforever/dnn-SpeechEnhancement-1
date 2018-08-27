import numpy as np
import utils

def featureExtraction(filePathFeat,AUDIO_dB_SPL,NFFT,STFT_OVERLAP,numBin,featMean,featStd):
	#  Feature Extraction function
	#   filePathFeat   : filepath name [string]
	#   AUDIO_dB_SPL   : dB SPL value to adjust SNR level
	#   NFFT           : Number of frequency bins
	#   STFT_OVERLAP   : frame overlab [0 - 1]
	#   numBin         : number of frequency bins you want to use. [0 Hz - ..]
    #   featMean       : features mean value for standardization
    #   featStd        : features standard deviation value for standardization

    x, fs = utils.wavToSamples(filePathFeat)
    x = utils.adjustSNR(x,AUDIO_dB_SPL)

    features,feature_phi,_,_ = utils.STFT(x,fs,NFFT,int(NFFT*STFT_OVERLAP))
    features = np.float32(features)
    features = np.log10(features + 1e-7)

    features = features - featMean
    features = features/featStd
    features = np.transpose(features)

    features = features[:,0:numBin]

    return features,feature_phi
