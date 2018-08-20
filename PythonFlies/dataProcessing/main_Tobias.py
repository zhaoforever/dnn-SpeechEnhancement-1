import sys
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import utils
import MCC_plot

import numpy as np
import matplotlib.pyplot as plt

### Dataset and feature extraction parameters ###
NFFT = 512
STFT_OVERLAP = 0.75

dataPath_true = 'C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/TIMIT_train_ref1/331_ref.wav'
dataPath_pred = 'C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/TIMIT_train_feat1/331_feat.wav'


MCC,segMCC,freq = MCC_plot.MCC_plot(dataPath_true,dataPath_pred,NFFT,STFT_OVERLAP)

plt.semilogx(freq,segMCC, linewidth=2,color='b')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude Correlation")
plt.title("MCC")
plt.grid(linestyle='--')
plt.show()
