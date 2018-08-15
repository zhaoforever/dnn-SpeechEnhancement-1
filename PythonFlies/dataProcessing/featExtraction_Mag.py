import numpy as np
import matplotlib.pyplot as plt
import os, sys
import time
from scipy import signal
from numpy import seterr,isneginf,isinf,isnan

import utils
#fileRoot = 'C:/Users/s123028/dataset8_MulitTfNoise/'
fileRoot = 'C:/Users/s123028/dataset1khz/'

######### Training labels 1 #############
data_root = data_root = fileRoot + 'TIMIT_train_ref1'
#data_root = data_root = fileRoot + 'trainLabels1'
nfft = 512
wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Training labels')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)


_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization

X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
#featMean = np.asarray(np.mean(X_abs,1))
#featMean = np.mean(X_abs)
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
#featStd = np.asarray(np.std(X_abs,1))
#featStd = np.std(X_abs)
X_abs = (X_abs.T/featStd).T


#trainingStats = [featMean,featStd]

#np.save('trainingStats', trainingStats)

#hist, bins = np.histogram(X_abs3,bins=32)
#plt.plot(bins[:-1],hist)

#plt.pcolormesh((X_abs[:,0:300]))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_label_train1', data_feat)

utils.dispBound(data_feat)







######### Training labels 2 #############
data_root = data_root = fileRoot + 'TIMIT_train_ref2'
#data_root = data_root = fileRoot + 'trainLabels2'

nfft = 512
wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Training labels')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)


_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization

X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T/featStd).T
print(np.isfinite(X_abs).all())


data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_label_train2', data_feat)










############# Training features 1 ###############
data_root = fileRoot + 'TIMIT_train_feat1'
#data_root = data_root = fileRoot + 'trainFeatures1'

#data_root = 'C:/Users/s123028/Master Thesis Data/dataset2/TIMIT_train_BCM'
nfft = 512

wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Training features')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)


_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization


X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T/featStd).T
print(np.isfinite(X_abs).all())

hist, bins = np.histogram(X_abs,bins=32)
plt.plot(bins[:-1],hist)

plt.pcolormesh((X_abs[:,500:1500]))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_feat_train1', data_feat)














############# Training features 2 ###############
data_root = fileRoot + 'TIMIT_train_feat2'
#data_root = data_root = fileRoot + 'trainFeatures2'

#data_root = 'C:/Users/s123028/Master Thesis Data/dataset2/TIMIT_train_BCM'
nfft = 512

wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Training features')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)


_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization

X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T/featStd).T
print(np.isfinite(X_abs).all())


#hist, bins = np.histogram(X_abs,bins=32)
#plt.plot(bins[:-1],hist)

#plt.imshow((X_abs[:,0:300]))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_feat_train2', data_feat)
















#utils.dispBound(data_feat)







######### Validation features #############
data_root = data_root = fileRoot + 'TIMIT_val_feat'
#data_root = data_root = fileRoot + 'valFeatures'

nfft = 512
wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Validation features')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)

_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization
X_abs = np.float32(X_abs)

X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T/featStd).T
print(np.isfinite(X_abs).all())

#hist, bins = np.histogram(X_abs,bins=32)
#plt.plot(bins[:-1],hist)

#plt.imshow((X_abs[:,0:300]))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_feat_val', data_feat)



######### Validationg labels #############
data_root = data_root = fileRoot + 'TIMIT_val_ref'
#data_root = data_root = fileRoot + 'valLabels'

nfft = 512
wavAppend = []

trainingStats = np.load("trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

print('Validation labels')
for file in os.listdir(data_root):
    file_path = data_root + '/' + file
    wavData, fs = utils.wavToSamples(file_path)
    wavAppend += list(wavData)

_, _, Zxx = signal.stft(wavAppend, fs, nperseg=nfft,noverlap=int(nfft*0.75),return_onesided=True)
del wavAppend

X_abs = np.abs(Zxx)
X_abs.shape
del Zxx
# Feature normalization
X_abs = np.float32(X_abs)

X_abs = np.log10(X_abs + 1e-7)
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T - featMean).T
print(np.isfinite(X_abs).all())
X_abs = (X_abs.T/featStd).T
print(np.isfinite(X_abs).all())


#plt.imshow((X_abs[:,0:300] ))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

data_feat = np.transpose(X_abs)
print(data_feat.shape)

np.save('data_label_val', data_feat)
