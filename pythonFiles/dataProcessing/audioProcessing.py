import sys
sys.path.append('C:/Users/mhp/Documents/DNN_Datasets/allRawRecordings/')
from scipy.io import wavfile
import scipy.signal as dsp
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import time
wavFilesPath = 'C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/testInput/'


##### BCM FEATURES #####
wavFilesAllNames = os.listdir(wavFilesPath)
wavBcmAllNames = []
for file in wavFilesAllNames:
    if 'bcm' in file:
        wavBcmAllNames.append(file)

#wavFilesAllNames = wavFilesAllNames[:10]
wavBcmAllNames = wavBcmAllNames[5:]
#wavBcmAllNames[0]
tic = time.time()
fileNum = 1
for file in wavBcmAllNames:
    print('File number: ' + str(fileNum) + ' Filename: ' + file)
    wavFullPathBCM = wavFilesPath + file
    wavFullPathREF = wavFilesPath + file.replace('bcm','ref')




    ### Read wav data to samples ###
    fs, wav_data_BCM = wavfile.read(wavFullPathBCM)
    wav_data_BCM = wav_data_BCM/32767
    #wav_data_BCM = wav_data_BCM[0:20000]
    fs, wav_data_REF = wavfile.read(wavFullPathREF)
    wav_data_REF = wav_data_REF/32767
    #wav_data_REF = wav_data_REF[0:20000]

     ### Filters audio data ###
    low_cutoff = 60
    high_cutoff = 6000
    wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
    b, a = dsp.butter(4, wn, 'band')
    wav_data_BCM = dsp.filtfilt(b,a,wav_data_BCM)
    ### Resamples audio to 12 kHz ###
    wav_data_BCM = dsp.resample(wav_data_BCM,int(wav_data_BCM.shape[0]/4))
    #plt.plot(wav_data_12kHz)
    wav_data_BCM = utils.adjustSNR(wav_data_BCM,60)

    low_cutoff = 20
    high_cutoff = 6000
    wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
    b, a = dsp.butter(4, wn, 'band')
    wav_data_REF = dsp.filtfilt(b,a,wav_data_REF)
    ### Resamples audio to 12 kHz ###
    wav_data_REF = dsp.resample(wav_data_REF,int(wav_data_REF.shape[0]/4))
    #plt.plot(wav_data_12kHz)
    wav_data_REF = utils.adjustSNR(wav_data_REF,60)



    ### Splits data into N chunks ###
    N = int(len(wav_data_BCM)/30)
    n = 1
    for i in range(0,len(wav_data_BCM),N):
        temp_wav_data_BCM = wav_data_BCM[i:i+N]
        temp_wav_data_REF = wav_data_REF[i:i+N]

        corr = np.correlate(temp_wav_data_REF,temp_wav_data_BCM,"full");
        delay = int(len(corr)/2) - np.argmax(corr)

        if delay < 0:
            zeros = np.zeros((-delay))
            temp_wav_data_BCM = np.concatenate((zeros,temp_wav_data_BCM))
            temp_wav_data_REF = np.concatenate((temp_wav_data_REF,zeros))
        else:
            zeros = np.zeros((delay))
            temp_wav_data_REF = np.concatenate((zeros,temp_wav_data_REF))
            temp_wav_data_BCM = np.concatenate((temp_wav_data_BCM,zeros))

        temp_wav_data_BCM =  temp_wav_data_BCM*32767
        temp_wav_data_BCM = temp_wav_data_BCM.astype(np.int16)

        temp_wav_data_REF =  temp_wav_data_REF*32767
        temp_wav_data_REF = temp_wav_data_REF.astype(np.int16)

        newWavName = file[:-7] + 'feat_' + str(n) + '.wav'
        wavfile.write('C:/Users/mhp/Documents/DNN_Datasets/processedRecordings/' + newWavName,12000,temp_wav_data_BCM)

        newWavName = file[:-7] + 'ref_' + str(n) + '.wav'
        wavfile.write('C:/Users/mhp/Documents/DNN_Datasets/processedRecordingsRef/' + newWavName,12000,temp_wav_data_REF)
        n += 1
    fileNum += 1
toc = time.time()

print(toc-tic)
#
#     ### Correct delay ###
#     corr = np.correlate(wav_data_REF,wav_data_BCM,"full");
#     delay = int(len(corr)/2) - np.argmax(corr)
#     zeros = np.zeros((delay))
#     wav_data_REF = np.concatenate((zeros,wav_data_REF))
#     wav_data_BCM = np.concatenate((wav_data_BCM,zeros))
#
# #plt.plot(wav_data_REF)
# #plt.plot(wav_data_BCM)
#
#
#
#
#
# plt.plot(wav_data_BCM)
#
#     ### Filters audio data ###
#     low_cutoff = 60
#     high_cutoff = 6000
#     wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
#     b, a = dsp.butter(4, wn, 'band')
#     wav_data_filt = dsp.filtfilt(b,a,wav_data)
#     #plt.plot(wav_data_filt)
#
#     ### Resamples audio to 12 kHz ###
#     wav_data_12kHz = dsp.resample(wav_data_filt,int(wav_data_filt.shape[0]/4))
#     #plt.plot(wav_data_12kHz)
#     wav_data_60dB = utils.adjustSNR(wav_data_12kHz,60)
#     ### Splits data into N chunks ###
#     N = int(len(wav_data_60dB)/30)
#     n = 1
#     for i in range(0,len(wav_data_60dB),N):
#         temp_wav_data = wav_data_60dB[i:i+N]
#         temp_wav_data =  temp_wav_data*32767
#         temp_wav_data = temp_wav_data.astype(np.int16)
#
#         newWavName = file[:-7] + 'feat_' + str(n) + '.wav'
#         wavfile.write('C:/Users/mhp/Documents/DNN_Datasets/processedRecordings/' + newWavName,12000,temp_wav_data)
#         n += 1
#         #plt.plot(temp_wav_data)
#         #print(i)
# toc = time.time()
#
# print(toc-tic)
#
# ##### REFERENCE RECORDINGS #####
# wavFilesAllNames = os.listdir(wavFilesPath)
# wavRefAllNames = []
# for file in wavFilesAllNames:
#     if 'ref' in file:
#         wavRefAllNames.append(file)
#
# #wavFilesAllNames = wavFilesAllNames[:10]
#
# tic = time.time()
# for file in wavRefAllNames:
#
#     wavFullPath = wavFilesPath + file
#
#     ### Read wav data to samples ###
#     fs, wav_data = wavfile.read(wavFullPath)
#     wav_data = wav_data/32767
#
#     #plt.plot(wav_data)
#
#     ### Filters audio data ###
#     low_cutoff = 20
#     high_cutoff = 6000
#     wn = [low_cutoff/(fs/2), high_cutoff/(fs/2)]
#     b, a = dsp.butter(4, wn, 'band')
#     wav_data_filt = dsp.filtfilt(b,a,wav_data)
#     #plt.plot(wav_data_filt)
#
#     ### Resamples audio to 12 kHz ###
#     wav_data_12kHz = dsp.resample(wav_data_filt,int(wav_data_filt.shape[0]/4))
#     #plt.plot(wav_data_12kHz)
#     wav_data_60dB = utils.adjustSNR(wav_data_12kHz,60)
#     ### Splits data into N chunks ###
#     N = int(len(wav_data_60dB)/30)
#     n = 1
#     for i in range(0,len(wav_data_60dB),N):
#         temp_wav_data = wav_data_60dB[i:i+N]
#         temp_wav_data =  temp_wav_data*32767
#         temp_wav_data = temp_wav_data.astype(np.int16)
#
#         newWavName = file[:-7] + 'ref_' + str(n) + '.wav'
#         wavfile.write('C:/Users/mhp/Documents/DNN_Datasets/processedRecordingsRef/' + newWavName,12000,temp_wav_data)
#         n += 1
#         #plt.plot(temp_wav_data)
#         #print(i)
# toc = time.time()
#
# print(toc-tic)
