import tensorflow as tf
import numpy as np
import FFN_Model_gridSearch


slim = tf.contrib.slim

#filePath = "/zhome/8b/5/77233/Documents/realRecs/"
#filePath = "C:/Users/TobiasToft/Documents/GitHub/ABE_Master_thesis/PythonFiles/"
#filePath = "C:/Users/s123023/ABE_Master_thesis/PythonFiles/Real_imag_feat/"
filePath = "C:/Users/s123028/dataset8_MulitTfNoise/"



dataSetSize = 100
###################  Loading dataset ###################
trainingFeature = np.load(filePath + 'data_feat_train1.npy')
#trainingFeature1 = np.load(filePath + 'data_feat_train1.npy')
#trainingFeature2 = np.load(filePath + 'data_feat_train2.npy')
#trainingFeature = np.concatenate((trainingFeature1,trainingFeature2),axis=0)
#del trainingFeature1
#del trainingFeature2
trainingFeature = np.float32(trainingFeature)
trainingFeature = trainingFeature[0:dataSetSize,:]
print("Training feature dataset uploaded: ", trainingFeature.shape)

trainingLabel = np.load(filePath + 'data_label_train1.npy')
#trainingLabel1 = np.load(filePath + 'data_label_train1.npy')
#trainingLabel2 = np.load(filePath + 'data_label_train2.npy')
#trainingLabel = np.concatenate((trainingLabel1,trainingLabel2),axis=0)
#del trainingLabel1
#del trainingLabel2
trainingLabel = np.float32(trainingLabel)
trainingLabel = trainingLabel[0:dataSetSize,:]
print("Training label dataset uploaded: ", trainingLabel.shape)

validationFeature = np.load(filePath + 'data_feat_val.npy')
validationFeature = np.float32(validationFeature)
validationFeature = validationFeature[0:dataSetSize//10,:]
print("Validation feature dataset uploaded: ", validationFeature.shape)

validationLabel = np.load(filePath + 'data_label_val.npy')
validationLabel = np.float32(validationLabel)
validationLabel = validationLabel[0:dataSetSize//10,:]
print("Validation label dataset uploaded: ", validationLabel.shape)


################### Dataset placeholder setup ###################
## Training placeholders
features_placeholder = tf.placeholder(trainingFeature.dtype, trainingFeature.shape)
labels_placeholder = tf.placeholder(trainingLabel.dtype, trainingLabel.shape)

# Converts the features and labels to tf.datasets
trainDataFeatures = tf.data.Dataset.from_tensor_slices(features_placeholder)
trainDataLabels = tf.data.Dataset.from_tensor_slices(labels_placeholder)

# Combines the features and labels in one Dataset
trainDataset = tf.data.Dataset.zip((trainDataFeatures, trainDataLabels))

## Validation placeholders
valFeatures_placeholder = tf.placeholder(validationFeature.dtype, validationFeature.shape)
valLabels_placeholder = tf.placeholder(validationLabel.dtype, validationLabel.shape)

# Converts the features and labels to tf.datasets
valDataFeatures = tf.data.Dataset.from_tensor_slices(valFeatures_placeholder)
valDataLabels = tf.data.Dataset.from_tensor_slices(valLabels_placeholder)

# Combines the features and labels in one Dataset
valDataset = tf.data.Dataset.zip((valDataFeatures, valDataLabels))




################### Dataset batch generator ###################
batchSize = 64
#trainDataset = trainDataset.shuffle(buffer_size=10000)
trainDataset = trainDataset.batch(batchSize) # Batch size
trainIterator = trainDataset.make_initializable_iterator()
next_trainFeat, next_trainLabel = trainIterator.get_next()
next_trainElement = (next_trainFeat,next_trainLabel)

#dataset = dataset.shuffle(buffer_size=10000)
valDataset = valDataset.batch(batchSize)
valIterator = valDataset.make_initializable_iterator()
next_valFeat, next_valLabel = valIterator.get_next()
next_valElement = (next_valFeat,next_valLabel)



############## FFN network model ##############
next_feat_pl = tf.placeholder(next_trainFeat.dtype,next_trainFeat.shape)
next_label_pl = tf.placeholder(next_trainLabel.dtype,next_trainLabel.shape)

NUM_CLASSES = int(np.size(next_trainLabel,axis=1))
print(NUM_CLASSES)

nLayers = [1, 2, 3, 4]
nUnits = [257, 512, 1024, 2048, 4096]
conigN = 1

for i in nLayers:
	for j in range(0,len(nUnits)):

		slimStack = [nUnits[j]]*i

		preds = FFN_Model_gridSearch.defineFFN(next_feat_pl,NUM_CLASSES,slimStack,conigN)
		conigN += 1



		############## Training Models ##############
		global_step = tf.Variable(0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

		## loss function scope:
		loss = tf.losses.mean_squared_error(next_label_pl,preds)


		## Training operation scope:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

		# applying the gradients
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

		############## Training loop ##############
		epoch_num = 5
		VALID_EVERY = 1

		train_loss = []

		val_loss = []

		stopCount = 10
		valStopping = np.ones(stopCount)*100
		stopIdx = 0
		trainingBool = True

		trainCount = 0
		resPreds = []

		model_variables = slim.get_model_variables() # Model variables for storing model
		saver = tf.train.Saver(model_variables)

		print('Training ...')
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#model_variables = slim.get_model_variables()
			#new_saver = tf.train.Saver(model_variables)
			#new_saver.restore(sess, tf.train.latest_checkpoint('./savedModels/'))
			for epoch in range(0,epoch_num):
				sess.run(trainIterator.initializer,
						feed_dict={features_placeholder: trainingFeature, labels_placeholder: trainingLabel})

				sess.run(valIterator.initializer,
						feed_dict={valFeatures_placeholder: validationFeature, valLabels_placeholder: validationLabel})

				while trainingBool:
					try:
						trainCount += 1
						## Training:
						next_feat,next_label = sess.run(next_trainElement)


						fetches_train = [train_op, loss,preds,global_step]
						feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label}

						# running the traning optimizer
						res_train = sess.run(fetches=fetches_train, feed_dict=feed_dict_train)

						train_loss.append(res_train[1])

						## Validation:
						if trainCount % VALID_EVERY == 0:
							try:
								next_feat,next_label = sess.run(next_valElement)
								fetches_val = [loss]
								feed_dict_val = {next_feat_pl: next_feat, next_label_pl: next_label}

								res_val = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)

								val_loss.append(res_val[0])
								#val_acc.append(res_train[2])

							except tf.errors.OutOfRangeError:

								sess.run(valIterator.initializer,
										feed_dict={valFeatures_placeholder: validationFeature, valLabels_placeholder: validationLabel})

								next_feat,next_label = sess.run(next_valElement)
								fetches_val = [loss]
								feed_dict_val = {next_feat_pl: next_feat, next_label_pl: next_label}

								res_val = sess.run(fetches=fetches_val, feed_dict=feed_dict_val)

								val_loss.append(res_val[0])

					except tf.errors.OutOfRangeError:
						print('End of epoch: ',epoch)
						val_loss = sum(val_loss) / float(len(val_loss))
						train_loss = sum(train_loss) / float(len(train_loss))
						print("Training loss: ", train_loss, " Validation loss: ", val_loss)
						#ts = time.time()
						#st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
						#print(st)
						#with open("logFile.txt",'a') as f:
							#print("Training loss: ", train_loss, "Validation loss: ", val_loss, file=f)
							#print("Timestamp: ", st, file=f)

						valStopping[stopIdx] = val_loss
						print(2*np.std(valStopping) + np.mean(valStopping))
						if val_loss > (2*np.std(valStopping) + np.mean(valStopping)):
							trainingBool = False
							epoch = epoch_num

						stopIdx = (stopIdx+1) % stopCount
						saveStr = './savedModels/my_test_model_e' + str(epoch) + 'i' + str(i)+ 'j' + str(nUnits[j]) + '.ckpt'
						saver.save(sess, saveStr)
						train_loss = []
						val_loss = []
						break

		print('Training done!')


#plt.imshow((np.transpose(trainingLabel[0:100,:])))
#res = res_train[2]
#plt.imshow((np.transpose(res)))
