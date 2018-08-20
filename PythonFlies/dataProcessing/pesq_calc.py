import sys
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import utils
import vqmetrics
import os
import numpy as np



def pesq_calc(dataPath_true,dataPath_pred):

    pesqScore = vqmetrics.pesq(dataPath_true, dataPath_pred)
    pesqScore = str(pesqScore)

    start_idx = pesqScore.find("'")

    pesqScore = np.asarray(pesqScore[start_idx+1:start_idx+6])
    pesqScore = np.float32(pesqScore)
    return pesqScore
