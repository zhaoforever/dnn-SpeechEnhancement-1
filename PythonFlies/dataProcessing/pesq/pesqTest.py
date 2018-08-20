import sys
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import utils
import vqmetrics
import os

#degraded = "C:/Users/s123028/realRecs/features_val/"
#reference = "C:/Users/s123028/realRecs/labels_val/"
dataPath_true = 'C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/TIMIT_train_ref1/331_ref.wav'
dataPath_pred = 'C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/TIMIT_train_feat1/331_feat.wav'


pesqScore = vqmetrics.pesq(dataPath_true, dataPath_pred)
print(pesqScore)
