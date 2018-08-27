import sys
sys.path.append('C:/Users/mhp/Documents/DNN_Datasets/recordings/')
from scipy.io import wavfile
import scipy.signal as dsp
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import time
wavFilesPath = 'C:/Users/mhp/Documents/DNN_Datasets/recordings/'


##### BCM FEATURES #####
wavFilesAllNames = os.listdir(wavFilesPath)
wavBcmAllNames = []
for file in wavFilesAllNames:
    if 'bcm' in file:
        wavBcmAllNames.append(file)

#wavFilesAllNames = wavFilesAllNames[:10]
len(wavBcmAllNames)
tic = time.time()
for file in wavBcmAllNames:

    wavFullPath = wavFilesPath + file

    ### Read wav data to samples ###
    fs, wav_data = wavfile.read(wavFullPath)
    wav_data = wav_data/32767

    #plt.plot(wav_data)

    ### Filters audio data ###
    low_cutoff = 60
    high_cutoff = 6000
    wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
    b, a = dsp.butter(4, wn, 'band')
    wav_data_filt = dsp.filtfilt(b,a,wav_data)
    #plt.plot(wav_data_filt)

    ### Resamples audio to 12 kHz ###
    wav_data_12kHz = dsp.resample(wav_data_filt,int(wav_data_filt.shape[0]/4))
    #plt.plot(wav_data_12kHz)
    wav_data_60dB = utils.adjustSNR(wav_data_12kHz,60)
    ### Splits data into N chunks ###
    N = int(len(wav_data_60dB)/30)
    n = 1
    for i in range(0,len(wav_data_60dB),N):
        temp_wav_data = wav_data_60dB[i:i+N]
        temp_wav_data =  temp_wav_data*32767
        temp_wav_data = temp_wav_data.astype(np.int16)

        newWavName = file[:-7] + 'feat_' + str(n) + '.wav'
        wavfile.write('./processedRecordings/' + newWavName,12000,temp_wav_data)
        n += 1
        #plt.plot(temp_wav_data)
        #print(i)
toc = time.time()

print(toc-tic)

##### REFERENCE RECORDINGS #####
wavFilesAllNames = os.listdir(wavFilesPath)
wavBcmAllNames = []
for file in wavFilesAllNames:
    if 'ref' in file:
        wavBcmAllNames.append(file)

#wavFilesAllNames = wavFilesAllNames[:10]

tic = time.time()
for file in wavBcmAllNames:

    wavFullPath = wavFilesPath + file

    ### Read wav data to samples ###
    fs, wav_data = wavfile.read(wavFullPath)
    wav_data = wav_data/32767

    #plt.plot(wav_data)

    ### Filters audio data ###
    low_cutoff = 60
    high_cutoff = 6000
    wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
    b, a = dsp.butter(4, high_cutoff/(fs/2), 'low')
    wav_data_filt = dsp.filtfilt(b,a,wav_data)
    #plt.plot(wav_data_filt)

    ### Resamples audio to 12 kHz ###
    wav_data_12kHz = dsp.resample(wav_data_filt,int(wav_data_filt.shape[0]/4))
    #plt.plot(wav_data_12kHz)
    wav_data_60dB = utils.adjustSNR(wav_data_12kHz,60)
    ### Splits data into N chunks ###
    N = int(len(wav_data_60dB)/30)
    n = 1
    for i in range(0,len(wav_data_60dB),N):
        temp_wav_data = wav_data_60dB[i:i+N]
        temp_wav_data =  temp_wav_data*32767
        temp_wav_data = temp_wav_data.astype(np.int16)

        newWavName = file[:-7] + 'ref_' + str(n) + '.wav'
        wavfile.write('./processedRecordingsRef/' + newWavName,12000,temp_wav_data)
        n += 1
        #plt.plot(temp_wav_data)
        #print(i)
toc = time.time()

print(toc-tic)
