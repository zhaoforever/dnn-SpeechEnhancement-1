import tensorflow as tf
#import utils
import numpy as np
#import matplotlib.pyplot as plt
#import datetime
#import time
#from numpy import seterr,isneginf,isnan,isfinite,isinf
#import ConvModel
import os
import random

#import tensorflow.contrib.slim as slim

slim = tf.contrib.slim


#filePath = "C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/"
#filePath = "/zhome/8b/5/77233/Documents/dataset7/"
#filePath2 = "/zhome/8b/5/77233/Documents/realRecs/"
#filePath = "C:/Users/s123023/ABE_Master_thesis/PythonFiles/Real_imag_feat/"
#filePath = "C:/Users/s123028/Master Thesis Data/dataset2/"
#filePath = "C:/Users/s123028/dataset4/"
filePath = "C:/Users/s123028/dataset7_MultiBcmTf/"


dataSetSize = 5000
print('Loading data...')
###################  Loading dataset ###################
trainingFeature = np.load(filePath + 'data_feat_train1.npy')
#trainingFeature1 = np.load(filePath + 'data_feat_train1.npy')
#trainingFeature2 = np.load(filePath + 'data_feat_train2.npy')
#trainingFeature = np.concatenate((trainingFeature1,trainingFeature2),axis=0)
#del trainingFeature1
#del trainingFeature2
trainingFeature = trainingFeature[0:dataSetSize,0:150]
trainingFeature = np.float32(trainingFeature)
print("Training feature dataset uploaded: ", trainingFeature.shape)

trainingLabel = np.load(filePath + 'data_label_train1.npy')
#trainingLabel1 = np.load(filePath + 'data_label_train1.npy')
#trainingLabel2 = np.load(filePath + 'data_label_train2.npy')
#trainingLabel = np.concatenate((trainingLabel1,trainingLabel2),axis=0)
#del trainingLabel1
#del trainingLabel2
trainingLabel = trainingLabel[0:dataSetSize,:]
trainingLabel = np.float32(trainingLabel)
print("Training label dataset uploaded: ", trainingLabel.shape)

validationFeature = np.load(filePath + 'data_feat_val.npy')
validationFeature = validationFeature[0:dataSetSize//10,0:150]
validationFeature = np.float32(validationFeature)
print("Validation feature dataset uploaded: ", validationFeature.shape)

validationLabel = np.load(filePath + 'data_label_val.npy')
validationLabel = validationLabel[0:dataSetSize//10,:]
validationLabel = np.float32(validationLabel)
print("Validation label dataset uploaded: ", validationLabel.shape)
print('Load complete!')

#hist, bins = np.histogram(trainingFeature,bins=32)
#plt.plot(bins[:-1],hist)


frameWidth = 10
nchannels = 1
batchSize = 32
trainingFeature.shape

trainingFeature = np.concatenate((np.repeat(trainingFeature[:1,:],frameWidth-1,axis=0),trainingFeature[:,:]),axis=0)

validationFeature = np.concatenate((np.repeat(validationFeature[:1,:],frameWidth-1,axis=0),validationFeature[:,:]),axis=0)


############## FFN network model ##############
NUM_CLASSES = int(trainingLabel.shape[1])
print(NUM_CLASSES)

next_feat_pl = tf.placeholder(tf.float32 ,[batchSize, frameWidth, trainingFeature.shape[1], nchannels])
next_label_pl = tf.placeholder(tf.float32 ,[batchSize, NUM_CLASSES])


next_feat_pl.shape


############## Training Models ##############

keepProb = tf.placeholder(tf.float32)
#preds = ConvModel.defineConv(next_feat_pl,NUM_CLASSES,keepProb)
with slim.arg_scope([slim.fully_connected,slim.conv2d],reuse=tf.AUTO_REUSE,activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1/1024))),tf.variable_scope('CNN'):
	conv1 = slim.conv2d(next_feat_pl, 32, [5,5],scope='conv1')

	pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

	conv2 = slim.conv2d(pool1, 64, [5,5],scope='conv2')

	pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

	x = slim.flatten(pool2)
	x = slim.fully_connected(x, num_outputs=2048,scope='fc1')

	x = tf.nn.dropout(x,keepProb)

	x = slim.fully_connected(x, num_outputs=2048,scope='fc2')

	x = tf.nn.dropout(x,keepProb)

	x = slim.fully_connected(x, num_outputs=2048,scope='fc3')

	x = tf.nn.dropout(x,keepProb)

	preds = slim.fully_connected(x, num_outputs=NUM_CLASSES,scope='fc_out')

## loss function scope:
meanLPC = np.asarray(np.load('meanLPC.npy')).T
loss = tf.losses.mean_squared_error(next_label_pl,preds,weights=meanLPC)
#loss = tf.losses.mean_squared_error(next_label_pl,preds)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001) #defining optimizer

    # applying the gradients
train_op = optimizer.minimize(loss=loss)


############## Training loop ##############
epoch_num = 50

train_loss = []
val_loss = []
val_loss_best = 100

stopCount = 10
bestCount = 0
valStopping = np.ones(stopCount)*100
stopIdx = 0
trainingBool = True
validationBool = True

trainCount = 0
valCount = 0

idx1Train = 0
idx2Train = idx1Train+frameWidth

idx1Val = 0
idx2Val = idx1Val+frameWidth

model_variables = slim.get_model_variables() # Model variables for storing model
saver = tf.train.Saver(model_variables)



print('Training ...')
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#model_variables = slim.get_model_variables()
	#new_saver = tf.train.Saver(model_variables)
	#new_saver.restore(sess, tf.train.latest_checkpoint('./savedModels/'))
	for epoch in range(0,epoch_num):
		train_loss = []
		val_loss = []

		#### Training ####
		while ((idx2Train+batchSize) <= trainingFeature.shape[0]) and (trainingBool):
			#try:
			trainCount += 1
			## Training:
			#print(trainCount)
			next_feat = np.empty((0,trainingFeature.shape[1]))
			next_label = np.empty((0))
			for n in range(0,batchSize):
				next_feat = np.concatenate((next_feat,trainingFeature[idx1Train:idx2Train,:]),axis=0)
				#print(next_feat.shape)

				next_label = np.concatenate((next_label,trainingLabel[idx1Train,:]),axis=0)
				#print(next_label.shape)
				idx1Train +=1
				idx2Train +=1

			next_feat = np.reshape(next_feat,[-1,frameWidth,trainingFeature.shape[1],1])
			next_label = np.reshape(next_label,[-1,trainingLabel.shape[1]])



			fetches_train = [train_op,loss]
			feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 0.75}

			# running the traning optimizer
			res_train = sess.run(fetches=fetches_train,
			feed_dict=feed_dict_train)

			train_loss.append(res_train[1])

			#except IndexError:
		trainingBool = False
		idx1Train = 0
		idx2Train = idx1Train+frameWidth

		#### Validation ####
		#while ((idx2Val+batchSize) <= validationFeature.shape[0]) & (validationBool):
		while (idx2Val+batchSize) <= validationFeature.shape[0] and (validationBool):

			next_feat = np.empty((0,validationFeature.shape[1]))
			next_label = np.empty((0))
			for n in range(0,batchSize):
				next_feat = np.concatenate((next_feat,validationFeature[idx1Val:idx2Val,:]),axis=0)
				#print(next_feat.shape)

				next_label = np.concatenate((next_label,validationLabel[idx1Val,:]),axis=0)
				#print(next_label.shape)
				idx1Val +=1
				idx2Val +=1

			next_feat = np.reshape(next_feat,[-1,frameWidth,validationFeature.shape[1],1])
			next_label = np.reshape(next_label,[-1,validationLabel.shape[1]])



			fetches_val = [loss]
			feed_dict_val = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: 1.0}

			# running the validation
			res_val = sess.run(fetches=fetches_val,
			feed_dict=feed_dict_val)

			val_loss.append(res_val[0])
			#print('res: ', res_val[0])

		idx1Val = 0
		idx2Val = idx1Val+frameWidth

		#### End of epoch operations ####
		#print(val_loss)
		val_loss = np.mean(val_loss)
		train_loss = np.mean(train_loss)

		print("Training loss: ", train_loss, " Validation loss: ", val_loss)

		printText = "\nEpoch: "+str(epoch)+"\nTraining loss: "+str(train_loss) +" Validation loss: "+str(val_loss)
		file = open("./savedModels/ModelResults.txt","a")
		file.write(printText)
		file.close()

		# Early stopping #
		print(val_loss_best)
		if val_loss < val_loss_best:

			val_loss_best = val_loss

			trainingBool = True
			validationBool = True

			bestCount = 0

			# Save model
			saveStr = './savedModels/my_test_model' + str(epoch) + '.ckpt'
			saver.save(sess, saveStr)

		elif bestCount < stopCount:
			trainingBool = True
			validationBool = True

			bestCount += 1
		else:
			trainingBool = False
			validationBool = False

		print('End of epoch: ',epoch)


print('Training done!')
