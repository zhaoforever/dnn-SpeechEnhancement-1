import tensorflow as tf

import numpy as np
import CNN_Model_gridSearch

slim = tf.contrib.slim

#filePath = 'C:/Users/TobiasToft/Desktop/multiSyncWithNoise_2/'
#filePath = '/zhome/1d/4/77309/Documents/MultiBCMTFWithNoise/'
#filePath = "/zhome/8b/5/77233/Documents/dataset7/"
#filePath = "/zhome/8b/5/77233/Documents/dataset8/"
filePath = "C:/Users/s123028/dataset7_MultiBcmTf/"



dataSetSize = 100
print('Loading data...')
###################  Loading dataset ###################
trainingFeature1 = np.load(filePath + 'data_feat_train1.npy')
trainingFeature1 = trainingFeature1[:dataSetSize,:150]
trainingFeature1 = np.float32(trainingFeature1)
print("Training feature dataset uploaded: ", trainingFeature1.shape)

trainingFeature2 = np.load(filePath + 'data_feat_train2.npy')
trainingFeature2 = trainingFeature2[:dataSetSize,:150]
trainingFeature2 = np.float32(trainingFeature2)
print("Training feature dataset uploaded: ", trainingFeature2.shape)

trainingLabel1 = np.load(filePath + 'data_label_train1.npy')
trainingLabel1 = trainingLabel1[:dataSetSize,:]
trainingLabel1 = np.float32(trainingLabel1)
print("Training label dataset uploaded: ", trainingLabel1.shape)

trainingLabel2 = np.load(filePath + 'data_label_train2.npy')
trainingLabel2 = trainingLabel2[0:dataSetSize,:]
trainingLabel2 = np.float32(trainingLabel2)
print("Training label dataset uploaded: ", trainingLabel2.shape)


validationFeature = np.load(filePath + 'data_feat_val.npy')
validationFeature = validationFeature[:dataSetSize//10,0:150]
validationFeature = np.float32(validationFeature)
print("Validation feature dataset uploaded: ", validationFeature.shape)

validationLabel = np.load(filePath + 'data_label_val.npy')
validationLabel = validationLabel[:dataSetSize//10,:]
validationLabel = np.float32(validationLabel)
print("Validation label dataset uploaded: ", validationLabel.shape)
print('Load complete!')

trainingFeature = np.concatenate((trainingFeature1,trainingFeature2),axis=0)
trainingLabel = np.concatenate((trainingLabel1,trainingLabel2),axis=0)


frameWidth = 10
nchannels = 1
batchSize = 8

trainingFeature = np.concatenate((np.repeat(trainingFeature[:1,:],frameWidth-1,axis=0),trainingFeature[:,:]),axis=0)
validationFeature = np.concatenate((np.repeat(validationFeature[:1,:],frameWidth-1,axis=0),validationFeature[:,:]),axis=0)


############## FFN network model ##############
NUM_CLASSES = int(trainingLabel.shape[1])
print(NUM_CLASSES)

next_feat_pl = tf.placeholder(tf.float32 ,[batchSize, frameWidth, trainingFeature.shape[1], nchannels])
next_label_pl = tf.placeholder(tf.float32 ,[batchSize, NUM_CLASSES])


convLayers = 1
L_rate = [0.001, 0.0001, 0.00001, 0.000001]
conigN = 1

for rate in L_rate:
	############## Training Models ##############
	global_step = tf.Variable(
		0, name='global_step', trainable=False,
		collections=[tf.GraphKeys.GLOBAL_VARIABLES,
					 tf.GraphKeys.GLOBAL_STEP])

	keepProb = tf.placeholder(tf.float32)

	preds = CNN_Model_gridSearch.defineConv(next_feat_pl,NUM_CLASSES,convLayers,conigN,keepProb)
	conigN += 1


	## loss function scope:
	#meanLPC = np.asarray(np.load('meanLPC.npy')).T
	#loss = tf.losses.mean_squared_error(next_label_pl,preds,weights=meanLPC)
	loss = tf.losses.mean_squared_error(next_label_pl,preds)
	optimizer = tf.train.AdamOptimizer(learning_rate=rate) #defining optimizer

	    # applying the gradients
	train_op = optimizer.minimize(loss=loss)


	############## Training loop ##############
	epoch_num = 250

	train_loss = []
	val_loss = []
	val_loss_best = 100

	stopCount = 10
	bestCount = 0

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
			if trainingBool:
				while ((idx2Train+batchSize) <= trainingFeature.shape[0]):
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
				while ((idx2Val+batchSize) <= validationFeature.shape[0]) and validationBool:

					#try:
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

				validationBool = False

				idx1Val = 0
				idx2Val = idx1Val+frameWidth

				#### End of epoch operations ####
				#print(val_loss)
				val_loss = np.mean(val_loss)
				train_loss = np.mean(train_loss)

				print(val_loss_best)
				print("Training loss: ", train_loss, " Validation loss: ", val_loss)

				# Early stopping #
				if val_loss < val_loss_best:
					trainingBool = True
					validationBool = True

					val_loss_best = val_loss

					bestCount = 0

				# Save model
					saveStr = './savedModels/my_test_model_e' + str(epoch) + 'R' + str(rate)+ 'L' + str(convLayers) + '.ckpt'
					saver.save(sess, saveStr)

				elif bestCount < stopCount:
					trainingBool = True
					validationBool = True

					bestCount += 1
				else:
					trainingBool = False
					validationBool = False


				print('End of epoch: ',epoch)
				printText = "\nEpoch: "+str(epoch)+"\nTraining loss: "+str(train_loss) +" Validation loss: "+str(val_loss)
				file = open("./savedModels/ModelResults.txt","a")
				file.write(printText)
				file.close()

	print('Training done!')
