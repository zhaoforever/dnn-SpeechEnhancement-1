import tensorflow as tf
import ConvModel
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
import sounddevice as sd
from scipy import signal
slim = tf.contrib.slim
from numpy import seterr,isneginf,isinf,isnan
from scipy.io import wavfile

import imp
import sys
sys.path.append("C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/")
import utils
utils = imp.reload(utils)


#datasetPath = 'C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/'
datasetPath = "C:/Users/s123028/dataset7_MultiBcmTf/"
#datasetPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
#datasetPath = "C:/Users/s123028/dataset8_NewNorm/"
#datasetPath = "C:/Users/s123028/dataset10_vctk/"
#datasetPath = "C:/Users/s123028/realRecs/"
trainingStats = np.load(datasetPath + "trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

#filePath = 'C:/Users/TobiasToft/Downloads/TIMIT_val_feat_BPfil/TIMIT_val_feat/942_feat.wav'
#filePath2 = 'C:/Users/TobiasToft/Downloads/TIMIT_val_feat_BPfil/TIMIT_val_ref/942_ref.wav'

filePath  = datasetPath + "TIMIT_val_feat/331_val_feat.wav"
filePath2 = datasetPath + "TIMIT_val_ref/331_val_ref.wav"

#filePath  = datasetPath + "val_feat/feat_p232_032.wav"
#filePath2 = datasetPath + "val_ref/ref_p232_032.wav"


#filePath = "C:/Users/s123028/dataset8_MulitTfNoise/TIMIT_train_feat1/331_feat.wav"
#filePath2 = "C:/Users/s123028/dataset8_MulitTfNoise/TIMIT_train_ref1/331_ref.wav"

#filePath = "C:/Users/s123028/realRecs/valFeatures/carl_01_bcm.wav"
#filePath2 = "C:/Users/s123028/realRecs/valLabels/carl_01_ref.wav"

#filePath = "C:/Users/s123028/dens_01_3k.wav"
#filePath2 = "C:/Users/s123028/dens_01_ref.wav"

#filePath = "C:/Users/Mikkel/ABE_Master_thesis/MatlabFiles/DANOK/carlTest.wav"
#filePath = "C:/Users/s123028/MUSHRA_wavFiles/carl_01_bcm.wav"

#filePath = "C:/Users/Mikkel/ABE_Master_thesis/MatlabFiles/DANOK/jose3CV_32f.wav"
#filePath = "C:/Users/Mikkel/ABE_Master_thesis/MatlabFiles/DANOK/wavesBCM_100/cons4_talk1.wav"
#filePath2 = "C:/Users/Mikkel/ABE_Master_thesis/MatlabFiles/DANOK/joseDANOK/CV_REF_CLEAN/BI_1_ref.wav"

x, fs = utils.wavToSamples(filePath)
x = x[0:int(fs*6)]

x_ref, fs2 = utils.wavToSamples(filePath2)
x_ref = x_ref[0:int(fs2*6)]

wL = 512

features,X_phi,t,f= utils.STFT(x,fs,wL,int(wL*0.75))
features.shape
X_phi.shape

labels,L_phi,t2,f2= utils.STFT(x_ref,fs2,wL,int(wL*0.75))
labels.shape

#### MODEL ####
features = np.float32(features)
labels = np.float32(labels)

features = np.log10(features + 1e-7)
print(np.isfinite(features).all())
features = (features.T - featMean).T
features = (features.T / featStd).T
print(np.isfinite(features).all())
features.shape

labels = np.log10(labels + 1e-7)
print(np.isfinite(labels).all())
labels = (labels.T - featMean).T
labels = (labels.T / featStd).T
print(np.isfinite(labels).all())
labels.shape
features = np.transpose(features)
labels = np.transpose(labels)

features_old = features
features = features[:,0:150]
features.shape



frameWidth = 10
nchannels = 1
batchSize = 1

features = np.concatenate((np.repeat(features[:1,:],frameWidth-1,axis=0),features[:,:]),axis=0)

features.shape

NUM_CLASSES = int(labels.shape[1])
print(NUM_CLASSES)

############## FFN network model ##############
next_feat_pl = tf.placeholder(tf.float32 ,[batchSize, frameWidth, features.shape[1], nchannels])
next_label_pl = tf.placeholder(tf.float32 ,[batchSize, NUM_CLASSES])

next_feat_pl.shape

#preds = ConvModel.defineConv(next_feat_pl,NUM_CLASSES)

epoch_num = 1

finalPreds = []

idx1 = 0
idx2 = idx1+frameWidth
testLoss = []

with tf.Session() as sess:
	#preds = ConvModel.defineConv(next_feat_pl,NUM_CLASSES)
	keepProb = tf.placeholder(tf.float32)
	preds = ConvModel.defineConv(next_feat_pl,NUM_CLASSES,keepProb)
	loss = tf.losses.mean_squared_error(next_label_pl,preds)
	sess.run(tf.global_variables_initializer())
	model_variables = slim.get_model_variables()
	#print(model_variables)
	new_saver = tf.train.Saver(model_variables)
	#new_saver = tf.train.import_meta_graph('./savedModels/my_test_model.ckpt.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./savedModels/'))
	for epoch in range(0,epoch_num):
		while idx2 <= features.shape[0]:
			try:

				## Training:
				#next_feat = features[idx1:idx2,:]

				#next_feat = np.reshape(next_feat,[1,frameWidth,features.shape[1],1])

				next_feat = np.empty((0,features.shape[1]))
				next_label = np.empty((0))
				#for n in range(0,batchSize-1):
				next_feat = np.concatenate((next_feat,features[idx1:idx2,:]),axis=0)
					#print(next_feat.shape)
				next_label = np.concatenate((next_label,labels[idx1,:]),axis=0)

				idx1 +=1
				idx2 +=1

				next_feat = np.reshape(next_feat,[-1,frameWidth,features.shape[1],1])

				next_label = np.reshape(next_label,[-1,labels.shape[1]])

				fetches_test = [preds,loss]
				feed_dict_test = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 1.0}

				# running the traning optimizer
				res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)


				finalPreds.append(res_test[0])
				testLoss.append(res_test[1])

			except IndexError:

				break
testLoss =   np.mean(testLoss)
print("Test loss: ", testLoss)
finalPreds0 = np.transpose(np.asarray(finalPreds)[:,-1,:])
print('Training done!')


#chkp.print_tensors_in_checkpoint_file("./savedModels/my_test_model11.ckpt", tensor_name='', all_tensors=False,all_tensor_names=False)

#print(model_variables)

#from tensorflow.python import pywrap_tensorflow

#reader = pywrap_tensorflow.NewCheckpointReader("./savedModels/my_test_model99.ckpt")
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in sorted(var_to_shape_map):
#    print("tensor_name: ", key)
#    print(reader.get_tensor(key).shape)



plt.pcolormesh(finalPreds0)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


plt.pcolormesh(np.transpose(features_old))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


plt.pcolormesh(np.transpose(labels))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


hist, bins = np.histogram(finalPreds0,bins=32)
plt.plot(bins[:-1],hist)



#hist, bins = np.histogram(finalPreds,bins=32)
#plt.plot(bins[:-1],hist)

#hist, bins = np.histogram(1/alpha*np.transpose(features_old),bins=32)
#plt.plot(bins[:-1],hist)


alpha = 1.3
finalPreds = np.copy(finalPreds0)
finalPredsExp = alpha*np.copy(finalPreds0)


#finalPreds[np.where(finalPreds < -1)] = -100
#finalPreds = finalPreds + np.mean(finalPreds)
#finalPreds = (1/alpha*np.transpose(features_old)+alpha*finalPreds0)/alpha

#alpha = 0.2
#finalPreds = (1.0-alpha)*np.transpose(features_old)+alpha*finalPreds0

finalPreds1 = (finalPredsExp.T * featStd).T
finalPreds2 = (finalPreds1.T + featMean).T
finalPredsExp = np.copy(finalPreds2)
finalPreds3 = 10**(finalPreds2)-1e-7
#finalPreds3[:25,:] = 0
y = utils.ISTFT(finalPreds3,X_phi,fs,wL,int(wL*0.75))
y_Exp = y/np.max(np.abs(y))

finalPreds1 = (finalPreds.T * featStd).T
finalPreds2 = (finalPreds1.T + featMean).T
finalPreds3 = 10**(finalPreds2)-1e-7

#finalPreds3[:25,:] = 0
y = utils.ISTFT(finalPreds3,X_phi,fs,wL,int(wL*0.75))
y = y/np.max(np.abs(y))

finalPreds3.shape
X_phi.shape



features1 = (np.transpose(features_old).T * featStd).T
features2 = (features1.T + featMean).T
features3 = 10**(features2)-1e-7

x2 = utils.ISTFT(features3,X_phi,fs,wL,int(wL*0.75))
x2 = x2/np.max(np.abs(x2))

labels1 = (np.transpose(labels).T * featStd).T
labels2 = (labels1.T + featMean).T


labels3 = 10**(labels2)-1e-7
xRef2 = utils.ISTFT(labels3,L_phi,fs2,wL,int(wL*0.75))
xRef2 = xRef2/np.max(np.abs(xRef2))


hist, bins1 = np.histogram(finalPreds2,bins=32)
histExp, bins2 = np.histogram(finalPredsExp,bins=32)
histRef, bins3 = np.histogram(labels2,bins=32)
plt.plot(bins1[:-1],hist,bins2[:-1],histExp,bins3[:-1],histRef)
plt.plot(bins2[:-1],histExp)


x_ref.shape
y.shape

print(np.mean(np.sqrt((y_Exp-xRef2)**2)))
print(np.mean(np.sqrt((y-xRef2)**2)))
print(np.mean(np.sqrt((x2-xRef2)**2)))


yWav = y_Exp/np.max(np.abs(y_Exp))
yRms = np.sqrt(np.mean(np.square(yWav)))
yWav = yWav*10**((60-20*np.log10(yRms/2e-5))/20);
yWav_Exp= np.float32(yWav)

yWav = y/np.max(np.abs(y))
yRms = np.sqrt(np.mean(np.square(yWav)))
yWav = yWav*10**((60-20*np.log10(yRms/2e-5))/20);
yWav= np.float32(yWav)

x2Wav = x2/np.max(np.abs(x2))
x2Rms = np.sqrt(np.mean(np.square(x2Wav)))
x2Wav = x2Wav*10**((60-20*np.log10(x2Rms/2e-5))/20);
x2Wav = np.float32(x2Wav)

xRef2Wav = xRef2/np.max(np.abs(xRef2))
refRms = np.sqrt(np.mean(np.square(xRef2Wav)))
xRef2Wav = xRef2Wav*10**((60-20*np.log10(refRms/2e-5))/20);
xRef2Wav = np.float32(xRef2Wav)

t = np.linspace(0,np.size(yWav)/fs,np.size(yWav))
plt.plot(t,yWav_Exp)

plt.plot(t,yWav)
plt.plot(x2Wav)
plt.plot(xRef2Wav)


#a = 0.5
#yWav = (1.0-a)*x2Wav + a*yWav
yWav = np.asarray([yWav,yWav],dtype=np.float32).T

x2Wav = np.asarray([x2Wav,x2Wav],dtype=np.float32).T

xRef2Wav = np.asarray([xRef2Wav,xRef2Wav],dtype=np.float32).T


sd.play(yWav,fs)
sd.play(x2Wav,fs)
#sd.play(xRef2Wav,fs2)

predPath = "./wavPredictionFiles/"
wavfile.write((predPath + 'featTestMultBCM.wav'),fs,x2Wav)
wavfile.write((predPath + 'refTestMultBCM.wav'),fs,xRef2Wav)
wavfile.write((predPath + 'predTestMultBCM.wav'),fs,yWav)
wavfile.write((predPath + 'ExpTestMultBCM.wav'),fs,yWav_Exp)
