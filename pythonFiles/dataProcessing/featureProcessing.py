import numpy as np
import utils
from scipy.io import wavfile

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

def features2samples(features_abs,features_phi,featMean,featStd,fs,NFFT,STFT_OVERLAP):

    # features_abs  : Predictions from the network - shape = (FFT Bins, Time frames)
    # features_phi  : Original phase from the feature extraction - shape = (FFT Bins, Time frames)

    if not features_abs.shape == features_phi.shape:
        fillFreqs = np.full((features_phi.shape[0] - features_abs.shape[0],features_phi.shape[1]),np.min(features_abs))
        features_abs = np.concatenate((features_abs,fillFreqs),axis=0)

    features_abs = (features_abs * featStd)
    features_abs = (features_abs + featMean)
    features_abs = 10**(features_abs)-1e-7

    y = utils.ISTFT(features_abs,features_phi,fs,NFFT,int(NFFT*STFT_OVERLAP))
    y = utils.adjustSNR(y,60)
    y = np.float32(y)

    return y

def samples2wav(y,fs,filePathAndName):

    y =  y*32767
    y = y.astype(np.int16)
    wavfile.write(filePathAndName,fs,y)


    return
