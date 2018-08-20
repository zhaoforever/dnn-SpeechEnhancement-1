import sys
#sys.path.append("C:/Users/Mikkel/Google Drev/Invisio/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import FFNModelTF
import os
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from scipy.io import wavfile
import random
import imp
import matplotlib.pyplot as plt
import utils
import FeatureExtraction
utils = imp.reload(utils)

tf.reset_default_graph()

## Hyper- and model parameters ###
MAX_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
KEEP_PROB_TRAIN = 0.75
KEEP_PROB_VAL = 1.0

### Dataset and feature extraction parameters ###
DATASET_SIZE_TRAIN = 20
DATASET_SIZE_VAL = 2
NFFT = 512
NUMBER_BINS = 128
STFT_OVERLAP = 0.75
#NUM_CLASSES = int(NFFT/2+1)
NUM_CLASSES = NUMBER_BINS
AUDIO_dB_SPL = 60

### Early stopping criteria ###
STOP_COUNT = 10

### Path to dataset ###
#dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
feat_root_train = dataPath + "TIMIT_train_feat1/"
label_root_train = dataPath + "TIMIT_train_ref1/"

feat_root_val = dataPath + 'TIMIT_val_feat/'
label_root_val  = dataPath + 'TIMIT_val_ref/'

trainingStats = np.load(dataPath + "trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

### Model placeholders ###
next_feat_pl = tf.placeholder(tf.float32,[None,NUM_CLASSES],name='next_feat_pl')
next_label_pl=tf.placeholder(tf.float32,[None,NUM_CLASSES],name='next_label_pl')

keepProb = tf.placeholder(tf.float32,name='keepProb')

### Model definition ###
preds = FFNModelTF.defineFFN(next_feat_pl,NUM_CLASSES,keepProb)

### Optimizer ###
loss = tf.losses.mean_squared_error(next_label_pl,preds)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.minimize(loss=loss)

### Summaries ###
with tf.name_scope('performance'):
	loss_sum_train = tf.placeholder(tf.float32,shape=None,name='loss_summary_train')
	tf_loss_summary_train = tf.summary.scalar('loss_train', loss_sum_train)

	loss_sum_val = tf.placeholder(tf.float32,shape=None,name='loss_summary_val')
	tf_loss_summary_val = tf.summary.scalar('loss_val', loss_sum_val)

performance_summaries = tf.summary.merge([tf_loss_summary_train,tf_loss_summary_val])

for g,v in grads_and_vars:
	if 'out' in v.name and 'kernel' in v.name:
		with tf.name_scope('gradients'):
			last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
			gradnorm_summary = tf.summary.scalar('grad_norm',last_grad_norm)
			break

with tf.name_scope('tb_images'):
	tb_image = tf.placeholder(tf.float32,shape=None,name='tb_image')
	image_summary_op = tf.summary.image('images',tb_image, 1)

### Model variable initializer ###
train_loss = []
train_loss_mean = []
val_loss = []
val_loss_mean = []
val_loss_best = 1000 # large initial value
bestCount = 0
trainingBool = True
validationBool = True

epochCount = 0
allFilesTrain = os.listdir(feat_root_train)
allFilesTrain = allFilesTrain[0:DATASET_SIZE_TRAIN]

allFilesVal = os.listdir(feat_root_val)
allFilesVal = allFilesVal[0:DATASET_SIZE_VAL]

iter = 0
finalPreds = np.empty((0,NUM_CLASSES))

allEventFiles = os.listdir('./logs/')
for file in allEventFiles:
	os.remove('./logs/'+file)

print('Training...')
with tf.Session() as sess:
	#os.remove('./logs/')
	writer = tf.summary.FileWriter('logs', sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for epoch in range(0,MAX_EPOCHS):
		iter = 0
		if trainingBool:
			train_loss_mean = []
			val_loss_mean = []

			random.shuffle(allFilesTrain)
			#random.shuffle(allFilesVal)
			for file in allFilesTrain:
				filePathFeat = feat_root_train + '/' + file
				filePathLabel = label_root_train + '/' + file

				if epochCount == MAX_EPOCHS:
					break

				features,features_phi = FeatureExtraction.FeatureExtraction(filePathFeat,AUDIO_dB_SPL,NFFT,STFT_OVERLAP,NUMBER_BINS,featMean,featStd)
				labels,labels_phi = FeatureExtraction.FeatureExtraction(filePathLabel.replace('feat','ref'),AUDIO_dB_SPL,NFFT,STFT_OVERLAP,NUMBER_BINS,featMean,featStd)

				idx1Train = 0
				idx2Train = idx1Train+BATCH_SIZE


				while ((idx1Train+BATCH_SIZE) <= features.shape[0]) and (trainingBool):


					## Training:
					next_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,BATCH_SIZE):
						next_feat = np.concatenate((next_feat,features[idx1Train,:]),axis=0)

						next_label = np.concatenate((next_label,labels[idx1Train,:]),axis=0)
						idx1Train +=1
						idx2Train +=1

					next_feat = np.reshape(next_feat,[-1,features.shape[1]])
					next_label = np.reshape(next_label,[-1,labels.shape[1]])


					if iter == 0:
						fetches_train = [train_op,loss,gradnorm_summary]
						feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: KEEP_PROB_TRAIN}

						# running the traning optimizer
						res_train = sess.run(fetches=fetches_train,
						feed_dict=feed_dict_train)

						writer.add_summary(res_train[2], epoch)
					else:
						fetches_train = [train_op,loss]
						feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: KEEP_PROB_TRAIN}

						# running the traning optimizer
						res_train = sess.run(fetches=fetches_train,
						feed_dict=feed_dict_train)

					train_loss.append(res_train[1])
					iter =+ 1

				train_loss_mean.append(np.mean(train_loss))
				train_loss = []
	 		#### Validation ####
			for file in allFilesVal:
				filePathFeat_val = feat_root_val + '/' + file
				filePathRef_val = label_root_val + '/' + file

				features_val,features_val_phi = FeatureExtraction.FeatureExtraction(filePathFeat_val,AUDIO_dB_SPL,NFFT,STFT_OVERLAP,NUMBER_BINS,featMean,featStd)
				labels_val,labels_val_phi = FeatureExtraction.FeatureExtraction(filePathRef_val.replace('feat','ref'),AUDIO_dB_SPL,NFFT,STFT_OVERLAP,NUMBER_BINS,featMean,featStd)

				idx1Val = 0
				idx2Val = idx1Val+BATCH_SIZE


				while ((idx2Val+BATCH_SIZE) <= features_val.shape[0]) and (validationBool):


					## Validating:
					next_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,BATCH_SIZE):
						next_feat = np.concatenate((next_feat,features_val[idx1Val,:]),axis=0)

						next_label = np.concatenate((next_label,labels_val[idx1Val,:]),axis=0)
						idx1Val +=1
						idx2Val +=1

					next_feat = np.reshape(next_feat,[-1,features_val.shape[1]])
					next_label = np.reshape(next_label,[-1,labels_val.shape[1]])




					fetches_Val = [loss,preds]
					feed_dict_Val = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: KEEP_PROB_VAL}


					# running the validating run
					res_Val = sess.run(fetches=fetches_Val,
					feed_dict=feed_dict_Val)

					val_loss.append(res_Val[0])
					finalPreds = np.concatenate((finalPreds,res_Val[1]),axis=0)
				val_loss_mean.append(np.mean(val_loss))
				val_loss = []


			print('End of epoch: ',epoch+1)
			epochCount += 1

			### End of epoch rutines ###
			train_loss_mean = np.mean(train_loss_mean)
			val_loss_mean = np.mean(val_loss_mean)

			print("Training loss: ", train_loss_mean, " Validation loss: ", val_loss_mean)

			## Loss summary to
			summ = sess.run(performance_summaries,feed_dict={loss_sum_train: train_loss_mean, loss_sum_val: val_loss_mean})
			writer.add_summary(summ, epoch)


			finalPreds0 = np.flipud(finalPreds.T)
			finalPreds = np.empty((0,NUM_CLASSES))
			image_summary = sess.run(image_summary_op, feed_dict={tb_image: np.reshape(finalPreds0, [-1, finalPreds0.shape[0],finalPreds0.shape[1], 1])})
			writer.add_summary(image_summary, epoch)

			### Early stopping ###
			if val_loss_mean < val_loss_best:

				val_loss_best = val_loss_mean

				trainingBool = True
				validationBool = True

				bestCount = 0

				# Save model
				saveStr = './savedModelsWav/my_test_model' + str(epoch) + '.ckpt'
				saver.save(sess, saveStr)

			elif bestCount < STOP_COUNT:
				trainingBool = True
				validationBool = True

				bestCount += 1
			else:
				trainingBool = False
				validationBool = False

	print('Training done!')




plt.pcolormesh(features.T)
#plt.ylim((0, 1000))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
