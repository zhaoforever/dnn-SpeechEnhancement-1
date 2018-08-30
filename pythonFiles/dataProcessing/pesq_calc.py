import sys
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import utils
import vqmetrics
import os
import numpy as np
import featureProcessing
import scipy.signal as dsp



def pesq_calc(dataPath_true,dataPath_pred):

    tempPathRef = "./tempEvaluationFiles/tempPesqFileRef.wav"
    tempPathDeg = "./tempEvaluationFiles/tempPesqFileDeg.wav"

    fs16 = 16000



    reference, fs = utils.wavToSamples(dataPath_true)
    #lenRef = len(reference)
    #nextPow2 = 2**(lenRef - 1).bit_length()
    #reference = np.concatenate((reference,np.zeros(nextPow2-lenRef)))
    #lenRef = len(reference)
    #newN = int(lenRef*fs16/fs)
    #reference = dsp.resample(reference,newN)
    reference = dsp.resample_poly(reference,fs16,fs)
    featureProcessing.samples2wav(reference,fs16,tempPathRef)

    degraded, fs = utils.wavToSamples(dataPath_pred)
    # lenDeg = len(degraded)
    # nextPow2 = 2**(lenDeg - 1).bit_length()
    # degraded = np.concatenate((degraded,np.zeros(nextPow2-lenDeg)))
    # lenDeg = len(degraded)
    # newN = int(lenDeg*fs16/fs)
    # degraded = dsp.resample(degraded,newN)
    degraded = dsp.resample_poly(degraded,fs16,fs)
    featureProcessing.samples2wav(degraded,fs16,tempPathDeg)

    pesqScore = vqmetrics.pesq(tempPathRef, tempPathDeg)
    pesqScore = str(pesqScore)

    start_idx = pesqScore.find("'")

    pesqScore = np.asarray(pesqScore[start_idx+1:start_idx+6])
    pesqScore = np.float32(pesqScore)
    return pesqScore
