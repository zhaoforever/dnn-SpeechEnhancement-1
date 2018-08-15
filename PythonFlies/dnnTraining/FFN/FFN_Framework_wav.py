import tensorflow as tf
import FFNModelTF
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
# import sounddevice as sd
#from scipy import signal
from scipy.io import wavfile
import random
import imp
#
tf.reset_default_graph()
sys.path.append("C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/")
import utils
utils = imp.reload(utils)
#
# slim = tf.contrib.slim
#
dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
feat_root = dataPath + "TIMIT_train_feat1/"
label_root = dataPath + "TIMIT_train_ref1/"
#
# #data_root = "C:/Users/s123028/realRecs/features_val"
# #ref_root = "C:/Users/s123028/realRecs/labels_val"
#
batchSize = 32
#
# #dataPath = 'C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/realBCMRecLP_features/'
# #data_root = 'C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/realBCMRecLP_features/valFeatures'
# #ref_root = 'C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/realBCMRecLP_features/valLabels'
#
# dataPath = 'C:/Users/s123028/dataset8_MulitTfNoise/'
# data_root_Train = dataPath + 'TIMIT_train_feat1/'
# ref_root_Train  = dataPath + 'TIMIT_train_ref1/'
#
data_root_Val = dataPath + 'TIMIT_val_feat/'
ref_root_Val  = dataPath + 'TIMIT_val_ref/'
#
trainingStats = np.load(dataPath + "trainingStats.npy")
#
featMean = trainingStats[0]
featStd = trainingStats[1]
#
NUM_CLASSES = 257
#next_feat_pl = tf.placeholder(tf.float32 ,[batchSize, frameWidth, 257, nchannels],name='next_feat_pl')
#next_label_pl = tf.placeholder(tf.float32 ,[batchSize,

next_feat_pl = tf.placeholder(tf.float32,[batchSize,NUM_CLASSES],name='next_feat_pl')
next_label_pl=tf.placeholder(tf.float32,[batchSize,NUM_CLASSES],name='next_label_pl')

org_feat_pl = tf.placeholder(tf.float32 ,[batchSize,NUM_CLASSES],name='org_feat_pl')
#
#
keepProb = tf.placeholder(tf.float32,name='keepProb')
#
preds = FFNModelTF.defineFFN(next_feat_pl,NUM_CLASSES,keepProb)
#
# alpha = 2.0
# preds = (1.0/alpha*np.transpose(org_feat_pl)+alpha*preds)/alpha

# Train ops
loss = tf.losses.mean_squared_error(next_label_pl,preds)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001) #defining optimizer
train_op = optimizer.minimize(loss=loss)


train_loss = []
train_loss_mean = []
val_loss = []
val_loss_mean = []
val_loss_best = 100
bestCount = 0
stopCount = 20
trainingBool = True
validationBool = True
#
maxEpochs = 100
#
epochCount = 0
allFilesTrain = os.listdir(feat_root)
random.shuffle(allFilesTrain)
allFilesTrain = allFilesTrain[0:10]
#
allFilesVal = os.listdir(label_root)
random.shuffle(allFilesVal)
allFilesVal = allFilesVal[0:1]
#
#random.shuffle(allFiles)
lossSum = tf.placeholder(tf.float32,shape=None,name='loss_summary')
tf_loss_summary = tf.summary.scalar('lossSum', lossSum)

print('Training...')
with tf.Session() as sess:
	writer = tf.summary.FileWriter('logs', sess.graph)
	sess.run(tf.global_variables_initializer())
# 	model_variables = slim.get_model_variables()
	saver = tf.train.Saver()
	for epoch in range(0,maxEpochs):
		if trainingBool:
			train_loss_mean = []
			val_loss_mean = []
	#
			random.shuffle(allFilesTrain)
			#random.shuffle(allFilesVal)
			for file in allFilesTrain:
				filePathFeat = feat_root + '/' + file
				filePathLabel = label_root + '/' + file
	#
				if epochCount == maxEpochs:
					break
	#
				x, fs = utils.wavToSamples(filePathFeat)
				x = utils.adjustSNR(x,60)

				xRef, fs = utils.wavToSamples(filePathLabel.replace('feat','ref'))
				xRef = utils.adjustSNR(xRef,60)
	# 			#x = x[int(fs*1):int(fs*30)]
	# 			#xRef = xRef[int(fs*1):int(fs*30)]
	#
	#			features, X_phi =  utils.featureExtractMag512(x,fs,dataPath)
				#_, _, features = signal.stft(x, fs, nperseg=512,noverlap=int(512*0.75),return_onesided=True)
	#
				nfft = 512
				features,X_phi,_,_ = utils.STFT(x,fs,nfft,int(nfft*0.75))
				features = np.float32(features)
				#features = features[:150,:]
				features = np.log10(features + 1e-7)
				features = features - featMean
				features = features/featStd
	#
				features = np.transpose(features)
				features.shape

				labels,L_phi,_,_ = utils.STFT(xRef,fs,nfft,int(nfft*0.75))
				labels = np.float32(labels)
				#features = features[:150,:]
				labels = np.log10(labels + 1e-7)
				labels = labels - featMean
				labels = labels/featStd
		#
				labels = np.transpose(labels)
	# 			#labels, L_phi =  utils.featureExtractMag512(xRef,fs,dataPath)
	# 			_, _, labels = signal.stft(xRef, fs, nperseg=512,noverlap=int(512*0.75),return_onesided=True)
	#
	# 			labels = np.abs(labels)
	# 			labels = np.float32(labels)
	# 			labels = np.log10(labels + 1e-7)
	# 			labels = labels - featMean
	# 			labels = labels/featStd
	#
	# 			labels = np.transpose(labels)
	#
	# 			features = np.concatenate((np.repeat(features[:1,:],frameWidth-1,axis=0),features),axis=0)
	#
	#
				idx1Train = 0
				idx2Train = idx1Train+batchSize
	#
	#
				while ((idx1Train+batchSize) <= features.shape[0]) and (trainingBool):
	#
	#
					## Training:
					next_feat = np.empty((0))
					#org_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,batchSize):
						next_feat = np.concatenate((next_feat,features[idx1Train,:]),axis=0)
						#print(next_feat.shape)
						#org_feat = np.concatenate((org_feat,features[idx2Train,:]),axis=0)
	#
						next_label = np.concatenate((next_label,labels[idx1Train,:]),axis=0)
						#print(next_label.shape)
						idx1Train +=1
						idx2Train +=1
	#
					next_feat = np.reshape(next_feat,[-1,features.shape[1]])
	# 				org_feat = np.reshape(org_feat,[-1,features.shape[1]])
					next_label = np.reshape(next_label,[-1,labels.shape[1]])
	#
	#
					#print(idx2Train)
					fetches_train = [train_op,loss]
					feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 0.75}
	#
					# running the traning optimizer
					res_train = sess.run(fetches=fetches_train,
					feed_dict=feed_dict_train)
	#
					train_loss.append(res_train[1])
	#
				train_loss_mean.append(np.mean(train_loss))
				train_loss = []
	 		#### Validation ####
			for file in allFilesVal:
				filePath = data_root_Val + '/' + file
				filePath2 = ref_root_Val + '/' + file
	#
				x, fs = utils.wavToSamples(filePathFeat)
				x = utils.adjustSNR(x,60)

				xRef, fs = utils.wavToSamples(filePathLabel.replace('feat','ref'))
				xRef = utils.adjustSNR(xRef,60)
	#
				features,X_phi,_,_ = utils.STFT(x,fs,nfft,int(nfft*0.75))
				features = np.float32(features)
				#features = features[:150,:]
				features = np.log10(features + 1e-7)
				features = features - featMean
				features = features/featStd
				#
				features = np.transpose(features)
				features.shape

				labels,L_phi,_,_ = utils.STFT(xRef,fs,nfft,int(nfft*0.75))
				labels = np.float32(labels)
				#features = features[:150,:]
				labels = np.log10(labels + 1e-7)
				labels = labels - featMean
				labels = labels/featStd
				#
				labels = np.transpose(labels)
	#
				idx1Val = 0
				idx2Val = idx1Val+batchSize
	#
	#
				while ((idx2Val+batchSize) <= features.shape[0]) and (validationBool):
	#
	#
					## Validating:
					next_feat = np.empty((0))
					#org_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,batchSize):
						next_feat = np.concatenate((next_feat,features[idx1Val,:]),axis=0)
	# 					#print(next_feat.shape)
						#org_feat = np.concatenate((org_feat,features[idx2Val,:]),axis=0)
	#
						next_label = np.concatenate((next_label,labels[idx1Val,:]),axis=0)
	# 					#print(next_label.shape)
						idx1Val +=1
						idx2Val +=1
	#
					next_feat = np.reshape(next_feat,[-1,features.shape[1]])
	# 				org_feat = np.reshape(org_feat,[-1,features.shape[1]])
					next_label = np.reshape(next_label,[-1,labels.shape[1]])
	#
	#


					fetches_Val = [loss]
					feed_dict_Val = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 1.0}
	#

					# running the validating run
					res_Val = sess.run(fetches=fetches_Val,
					feed_dict=feed_dict_Val)
	#
					val_loss.append(res_Val[0])
	#
				val_loss_mean.append(np.mean(val_loss))
				val_loss = []
			print('End of epoch: ',epoch+1)
			epochCount += 1
	#
			train_loss_mean = np.mean(train_loss_mean)
			val_loss_mean = np.mean(val_loss_mean)

			print("Training loss: ", train_loss_mean, " Validation loss: ", val_loss_mean)
			#summary = sess.run(first_summary)
			summ = sess.run(tf_loss_summary,feed_dict={lossSum:val_loss_mean})
			writer.add_summary(summ, epoch)
	#
			if val_loss_mean < val_loss_best:
	#
				val_loss_best = val_loss_mean
	#
				trainingBool = True
				validationBool = True
	#
				bestCount = 0
	#
				# Save model
				saveStr = './savedModelsWav/my_test_model' + str(epoch) + '.ckpt'
				saver.save(sess, saveStr)
	#
			elif bestCount < stopCount:
				trainingBool = True
				validationBool = True
	#
				bestCount += 1
			else:
				trainingBool = False
				validationBool = False
	#
	#

	#
	#
	print('Training done!')
# #
