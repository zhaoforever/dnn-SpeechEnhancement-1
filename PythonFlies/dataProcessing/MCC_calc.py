import sys
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")

import numpy as np
import matplotlib.pyplot as plt
import utils


def MCC_calc(dataPath_true,dataPath_pred,NFFT):
    STFT_OVERLAP = 0.5

    feat_true, fs = utils.wavToSamples(dataPath_true)
    feat_true,_,_,_ = utils.STFT(feat_true,fs,NFFT,int(NFFT*STFT_OVERLAP))
    feat_true = np.float32(feat_true)
    #feat_true = np.array(feat_true)

    feat_pred, fs = utils.wavToSamples(dataPath_pred)
    feat_pred,_,_,_ = utils.STFT(feat_pred,fs,NFFT,int(NFFT*STFT_OVERLAP))
    feat_pred = np.float32(feat_pred)
    #feat_pred = np.array(feat_pred)



    # Calculate the spectral MCC
    xx_true = np.subtract(feat_true, np.mean(feat_true,1).reshape(feat_true.shape[0],1))
    denom_true = np.sqrt(np.sum(xx_true**2,1)).reshape(feat_true.shape[0],1)
    xx_true = np.divide(xx_true,denom_true)

    xx_pred = np.subtract(feat_pred, np.mean(feat_pred,1).reshape(feat_pred.shape[0],1))
    denom_pred = np.sqrt(np.sum(xx_pred**2,1)).reshape(feat_pred.shape[0],1)
    xx_pred = np.divide(xx_pred,denom_pred)

    segMCC = np.sum(np.multiply(xx_true,xx_pred),1)
    segMCC = segMCC.reshape(feat_pred.shape[0],1)
    MCC = np.mean(segMCC)

    freq = np.linspace(0,fs/2,np.size(segMCC,0));

#    plt.plot(freq,segMCC, linewidth=2)
#    plt.xlabel("Frequency [Hz]")
#    plt.ylabel("Magnitude Correlation")
#    plt.title("MCC")
#    plt.grid(linestyle='--')

    return MCC,segMCC,freq
