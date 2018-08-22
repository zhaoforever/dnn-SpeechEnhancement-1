import sys
sys.path.append("C:/Users/Mikkel/Desktop/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import os
import numpy as np
import random
import time
### Our functions ###
import FFN_Model_Cond_Dropout
import FeatureExtraction
import dataStatistics
import modelParameters as mp

tf.reset_default_graph()

# ## Hyper- and model parameters ###
# MAX_EPOCHS = 50
# BATCH_SIZE = 16
# LEARNING_RATE = 0.0001
# KEEP_PROB_TRAIN = 0.75
# KEEP_PROB_VAL = 1.0
# DECAYING_LEARNING_RATE = True
#
# ### Dataset and feature extraction parameters ###
# DATASET_SIZE_TRAIN = 5
# DATASET_SIZE_VAL = 2
# NUM_UNITS = 1024
# NFFT = 256
# NUMBER_BINS = int(NFFT/2+1)
# STFT_OVERLAP = 0.75
# NUM_CLASSES = NUMBER_BINS
# AUDIO_dB_SPL = 60
# BIN_WIZE = False
#
# ### Early stopping criteria ###
# STOP_COUNT = 15

### Path to dataset ###
dataPath = "C:/Users/s123028/dataset8_MulitTfNoise/"
#dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
feat_root_train = dataPath + "TIMIT_train_feat1/"
label_root_train = dataPath + "TIMIT_train_ref1/"

feat_root_val = dataPath + 'TIMIT_val_feat/'
label_root_val  = dataPath + 'TIMIT_val_ref/'

### Model placeholders ###
next_feat_pl = tf.placeholder(tf.float32,[None,mp.NUM_CLASSES],name='next_feat_pl')
next_label_pl=tf.placeholder(tf.float32,[None,mp.NUM_CLASSES],name='next_label_pl')

#learning_rate = tf.placeholder_with_default(LEARNING_RATE,shape=None,name='learning_rate')

keepProb = tf.placeholder_with_default(1.0,shape=None,name='keepProb')

is_train = tf.placeholder_with_default(False,shape=None,name="is_train")

### Model definition ###
preds = FFN_Model_Cond_Dropout.defineFFN(next_feat_pl,mp.NUM_UNITS,mp.NUM_CLASSES,keepProb,is_train)

### Optimizer ###
loss = tf.losses.mean_squared_error(next_label_pl,preds)

global_step = tf.Variable(0, name='global_step',trainable=False)

if mp.DECAYING_LEARNING_RATE:
	learning_rate = tf.train.exponential_decay(mp.LEARNING_RATE, global_step,
                                           100000, 0.96, staircase=True)
	rate_sum = tf.placeholder(tf.float32,shape=None,name='learning_rate_sum')
	tf_learning_rate_summary = tf.summary.scalar('Learning_Rate', rate_sum)

else:
	learning_rate = mp.LEARNING_RATE

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.minimize(loss=loss,global_step=global_step)

### Summaries ###
with tf.name_scope('performance'):
	loss_sum = tf.placeholder(tf.float32,shape=None,name='loss_summary')
	tf_loss_summary = tf.summary.scalar('loss', loss_sum)

	#loss_sum_val = tf.placeholder(tf.float32,shape=None,name='loss_summary_val')
	#tf_loss_summary_val = tf.summary.scalar('loss_val', loss_sum_val)

#performance_summaries = tf.summary.merge([tf_loss_summary_train,tf_loss_summary_val])

for g,v in grads_and_vars:
	if 'out' in v.name and 'kernel' in v.name:
		with tf.name_scope('gradients'):
			last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
			gradnorm_summary = tf.summary.scalar('grad_norm',last_grad_norm)
			break

with tf.name_scope('tb_images'):
	tb_image = tf.placeholder(tf.float32,shape=None,name='tb_image')
	image_summary_op_input  = tf.summary.image('images_input',tb_image, 1)
	image_summary_op_output = tf.summary.image('images_output',tb_image, 1)
	image_summary_op_target = tf.summary.image('images_target',tb_image, 1)

### Model variable initializer ###
train_loss = []
train_loss_mean = []
val_loss = []
val_loss_mean = []
val_loss_best = 1000 # large initial value
bestCount = 0
trainingBool = True
validationBool = True

#epochCount = 0
allFilesTrain = os.listdir(feat_root_train)
allFilesTrain = allFilesTrain[0:mp.DATASET_SIZE_TRAIN]

allFilesVal = os.listdir(feat_root_val)
allFilesVal = allFilesVal[0:mp.DATASET_SIZE_VAL]

### Removes old Tensorboard event files ###
allEventFiles = os.listdir('./logs/train/')
for file in allEventFiles:
	os.remove('./logs/train/'+file)

allEventFiles = os.listdir('./logs/val/')
for file in allEventFiles:
	os.remove('./logs/val/'+file)



### Data statistics ###
tic = time.time()
featMean, featStd = dataStatistics.calcMeanAndStd(label_root_train,mp.DATASET_SIZE_TRAIN,mp.NFFT,mp.STFT_OVERLAP,mp.BIN_WIZE)
toc = time.time()
print(np.round(toc-tic,2),"secs to calc data stats")

print('Training...')
with tf.Session() as sess:
	writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
	writer_val = tf.summary.FileWriter('logs/val/', sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	firstRun = True
	for epoch in range(1,mp.MAX_EPOCHS):
		tic = time.time()
		finalPreds = np.empty((0,mp.NUM_CLASSES))
		iter = 0
		if trainingBool:
			train_loss_mean = []
			val_loss_mean = []

			random.shuffle(allFilesTrain)
			for file in allFilesTrain:
				filePathFeat = feat_root_train + '/' + file
				filePathLabel = label_root_train + '/' + file

				features,_ = FeatureExtraction.FeatureExtraction(filePathFeat,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

				labels,_ = FeatureExtraction.FeatureExtraction(filePathLabel.replace('feat','ref'),mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

				idx1Train = 0

				while ((idx1Train+mp.BATCH_SIZE) <= features.shape[0]):

					### Training ###
					next_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,mp.BATCH_SIZE):
						next_feat = np.concatenate((next_feat,features[idx1Train,:]),axis=0)

						next_label = np.concatenate((next_label,labels[idx1Train,:]),axis=0)
						idx1Train +=1

					next_feat = np.reshape(next_feat,[-1,features.shape[1]])
					next_label = np.reshape(next_label,[-1,labels.shape[1]])


					if iter == 0:
						fetches_train = [train_op,loss,gradnorm_summary]
						feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: mp.KEEP_PROB_TRAIN,is_train: True}

						# running the traning optimizer
						res_train = sess.run(fetches=fetches_train,
						feed_dict=feed_dict_train)

						writer_train.add_summary(res_train[2], epoch)
					else:
						fetches_train = [train_op,loss]
						feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: mp.KEEP_PROB_TRAIN,is_train: True}

						# running the traning optimizer
						res_train = sess.run(fetches=fetches_train,
						feed_dict=feed_dict_train)

					train_loss.append(res_train[1])
					iter =+ 1

				train_loss_mean.append(np.mean(train_loss))
				train_loss = []

	 		#### Validation ####
			valFirstFile = True
			for file in allFilesVal:
				filePathFeat_val = feat_root_val + '/' + file
				filePathRef_val = label_root_val + '/' + file

				features_val,_ = FeatureExtraction.FeatureExtraction(filePathFeat_val,mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

				labels_val,_ = FeatureExtraction.FeatureExtraction(filePathRef_val.replace('feat','ref'),mp.AUDIO_dB_SPL,mp.NFFT,mp.STFT_OVERLAP,mp.NUMBER_BINS,featMean,featStd)

				idx1Val = 0

				while ((idx1Val+mp.BATCH_SIZE) <= features_val.shape[0]):

					## Validating:
					next_feat = np.empty((0))
					next_label = np.empty((0))
					for n in range(0,mp.BATCH_SIZE):
						next_feat = np.concatenate((next_feat,features_val[idx1Val,:]),axis=0)

						next_label = np.concatenate((next_label,labels_val[idx1Val,:]),axis=0)
						idx1Val +=1

					next_feat = np.reshape(next_feat,[-1,features_val.shape[1]])
					next_label = np.reshape(next_label,[-1,labels_val.shape[1]])

					fetches_Val = [loss,preds]
					feed_dict_Val = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 1.0}


					# running the validating run
					res_Val = sess.run(fetches=fetches_Val,
					feed_dict=feed_dict_Val)

					val_loss.append(res_Val[0])
					if valFirstFile:
						finalPreds = np.concatenate((finalPreds,res_Val[1]),axis=0)

				### Validation flag ###
				val_loss_mean.append(np.mean(val_loss))
				val_loss = []

				if firstRun:
					features_val_image = np.flipud(features_val.T)
					image_summary = sess.run(image_summary_op_input, feed_dict={tb_image: np.reshape(features_val_image, [-1, features_val_image.shape[0],features_val_image.shape[1], 1])})
					writer_val.add_summary(image_summary, epoch)

					labels_val_image = np.flipud(labels_val.T)
					image_summary = sess.run(image_summary_op_target, feed_dict={tb_image: np.reshape(labels_val_image, [-1, labels_val_image.shape[0],labels_val_image.shape[1], 1])})
					writer_val.add_summary(image_summary, epoch)
				firstRun = False
				valFirstFile = False

			print('End of epoch:',epoch)


			### End of epoch rutines ###
			train_loss_mean = np.mean(train_loss_mean)
			val_loss_mean = np.mean(val_loss_mean)
			print('Number of global steps: %s' % sess.run(tf.train.get_global_step()))
			print("Training loss: ", train_loss_mean, " Validation loss: ", val_loss_mean)

			### Loss summary to Tensorboard ###
			if mp.DECAYING_LEARNING_RATE:
				rate = np.float32(sess.run(learning_rate))
				summ = sess.run(tf_learning_rate_summary,feed_dict={rate_sum: rate})
				writer_train.add_summary(summ, epoch)
			else:
				rate = learning_rate

			summ = sess.run(tf_loss_summary,feed_dict={loss_sum: train_loss_mean})
			writer_train.add_summary(summ, epoch)

			summ = sess.run(tf_loss_summary,feed_dict={loss_sum: val_loss_mean})
			writer_val.add_summary(summ, epoch)


			finalPreds = np.flipud(finalPreds.T)
			image_summary = sess.run(image_summary_op_output, feed_dict={tb_image: np.reshape(finalPreds, [-1, finalPreds.shape[0],finalPreds.shape[1], 1])})
			writer_val.add_summary(image_summary, epoch)

			### Early stopping ###
			if val_loss_mean < val_loss_best:

				val_loss_best = val_loss_mean

				trainingBool = True
				validationBool = True

				bestCount = 0

				# Save model
				saveStr = './savedModelsWav/my_test_model' + str(epoch) + '.ckpt'
				saver.save(sess, saveStr)

			elif bestCount < mp.STOP_COUNT:
				trainingBool = True
				validationBool = True

				bestCount += 1
			else:
				trainingBool = False
				validationBool = False


			toc = time.time()
			print(np.round(toc-tic,2),"s")
			print()
		else:
			break
	writer_train.close()
	writer_val.close()
	print('Training done!')
