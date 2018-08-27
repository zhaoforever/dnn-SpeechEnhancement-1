import sys
sys.path.append("C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dataProcessing/")
#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import random
import time
import imp
### Our functions ###
import FFN_Model_Cond_Dropout
import featureProcessing
import dataStatistics
import modelParameters as mp
mp = imp.reload(mp)


### Path to dataset ###
dataPath = "C:/Users/mhp/Documents/DNN_Datasets/"
#dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
feat_root_train = dataPath + "TIMIT_train_feat1/"
label_root_train = dataPath + "TIMIT_train_ref1/"

feat_root_val = dataPath + 'TIMIT_val_feat/'
label_root_val  = dataPath + 'TIMIT_val_ref/'

### Hyperparameters for tunig ###
hp_learning_rate    = [0.0001, 0.00001, 0.000001]
#random.shuffle(hp_learning_rate)
hp_hidden_units     = [256, 512, 1024, 2048]
#random.shuffle(hp_hidden_units)
hp_nfft             = [128, 256, 512]
#random.shuffle(hp_nfft)
hp_batch_size       = [16, 32, 64, 128]
#random.shuffle(hp_batch_size)


hp_tuning_results = np.empty((5))

for rate in hp_learning_rate:
    for units in hp_hidden_units:
        for nfft in hp_nfft:
            for batch in hp_batch_size:
                tf.reset_default_graph()
                numberClasses = int(nfft/2+1)
                numberOfBins = numberClasses
                ### Model placeholders ###
                next_feat_pl = tf.placeholder(tf.float32,[None,numberClasses],name='next_feat_pl')
                next_label_pl=tf.placeholder(tf.float32,[None,numberClasses],name='next_label_pl')


                keepProb = tf.placeholder_with_default(1.0,shape=None,name='keepProb')

                is_train = tf.placeholder_with_default(False,shape=None,name="is_train")

                ### Model definition ###
                preds = FFN_Model_Cond_Dropout.defineFFN(next_feat_pl,units,numberClasses,keepProb,is_train)

                ### Optimizer ###
                loss = tf.losses.mean_squared_error(next_label_pl,preds)

                global_step = tf.Variable(0, name='global_step',trainable=False)

                if mp.DECAYING_LEARNING_RATE:
                    learning_rate = tf.train.exponential_decay(rate, global_step,
                                                           100000, 0.96, staircase=True)
                    rate_sum = tf.placeholder(tf.float32,shape=None,name='learning_rate_sum')
                    tf_learning_rate_summary = tf.summary.scalar('Learning_Rate', rate_sum)

                else:
                    learning_rate = rate

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                grads_and_vars = optimizer.compute_gradients(loss)
                train_op = optimizer.minimize(loss=loss,global_step=global_step)

                ### Summaries ###
                with tf.name_scope('performance'):
                    loss_sum = tf.placeholder(tf.float32,shape=None,name='loss_summary')
                    tf_loss_summary = tf.summary.scalar('loss', loss_sum)

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
                numFilesTrain = len(allFilesTrain)
                if mp.DATASET_SIZE_TRAIN > numFilesTrain:
                    mp.DATASET_SIZE_TRAIN = numFilesTrain
                allFilesTrain = allFilesTrain[0:mp.DATASET_SIZE_TRAIN]

                allFilesVal = os.listdir(feat_root_val)
                numFilesVal = len(allFilesVal)
                if mp.DATASET_SIZE_VAL > numFilesVal:
                    mp.DATASET_SIZE_VAL = numFilesVal
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
                featMean, featStd = dataStatistics.calcMeanAndStd(label_root_train,mp.DATASET_SIZE_TRAIN,nfft,mp.STFT_OVERLAP,mp.BIN_WIZE)
                toc = time.time()
                print(np.round(toc-tic,2),"secs to calc data stats")

                print('Training...')
                with tf.Session() as sess:
                    writer_train = tf.summary.FileWriter('logs/train/', sess.graph)
                    writer_val = tf.summary.FileWriter('logs/val/', sess.graph)
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(max_to_keep=1)

                    firstRun = True
                    for epoch in range(1,mp.MAX_EPOCHS+1):
                        tic = time.time()
                        finalPreds = np.empty((0,numberClasses))
                        iter = 0
                        if trainingBool:
                            train_loss_mean = []
                            val_loss_mean = []

                            random.shuffle(allFilesTrain)
                            for file in allFilesTrain:
                                filePathFeat = feat_root_train + file
                                filePathLabel = label_root_train + file

                                features,_ = featureProcessing.featureExtraction(filePathFeat,mp.AUDIO_dB_SPL,nfft,mp.STFT_OVERLAP,numberOfBins,featMean,featStd)

                                labels,_ = featureProcessing.featureExtraction(filePathLabel.replace('feat','ref'),mp.AUDIO_dB_SPL,nfft,mp.STFT_OVERLAP,numberOfBins,featMean,featStd)

                                idx1Train = 0

                                while ((idx1Train+batch) <= features.shape[0]):

                                    ### Training ###
                                    next_feat = np.empty((0))
                                    next_label = np.empty((0))
                                    for n in range(0,batch):
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

                                features_val,_ = featureProcessing.featureExtraction(filePathFeat_val,mp.AUDIO_dB_SPL,nfft,mp.STFT_OVERLAP,numberOfBins,featMean,featStd)

                                labels_val,_ = featureProcessing.featureExtraction(filePathRef_val.replace('feat','ref'),mp.AUDIO_dB_SPL,nfft,mp.STFT_OVERLAP,numberOfBins,featMean,featStd)

                                idx1Val = 0

                                while ((idx1Val+batch) <= features_val.shape[0]):

                                    ## Validating:
                                    next_feat = np.empty((0))
                                    next_label = np.empty((0))
                                    for n in range(0,batch):
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
                                saveStr = './savedModels_HP_Tune/hp_chechpoint' + str(rate) + '_' + str(units) + '_' + str(nfft) + '_' + str(batch) + '.ckpt'
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

                    #hp_tuning_results = np.concatenate((hp_tuning_results,np.asarray([rate,units,nfft,batch,val_loss_best])),axis=0)
                    if hp_tuning_results.all():
                        hp_tuning_results = np.row_stack((hp_tuning_results,np.asarray([rate,units,nfft,batch,val_loss_best])))
                    else:
                        hp_tuning_results = np.asarray([rate,units,nfft,batch,val_loss_best])





print('Training done!')


pandaArray = pd.DataFrame(hp_tuning_results)
pandaArray.columns = ['Learning Rate','Hidden Units','FFT Size','Batch Size','Validation Loss']
pandaArray = pandaArray.sort_values(by='Validation Loss')
resultsFilePath = 'hyperparameterTuningResults.xlsx'
pandaArray.to_excel(resultsFilePath,index=False)
