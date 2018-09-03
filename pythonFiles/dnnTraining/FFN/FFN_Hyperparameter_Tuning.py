import sys
#sys.path.append("/home/paperspace/Desktop/pythonFiles/dataProcessing/")
sys.path.append("C:/Users/mhp/Documents/GitHub/dnn-SpeechEnhancement/pythonFiles/dataProcessing/")
#sys.path.append("C:/Users/TobiasToft/Documents/GitHub/dnn-SpeechEnhancement/PythonFlies/dataProcessing/")
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import random
import time
import imp
import matplotlib.pyplot as plt
import scipy.signal as dsp
### Our functions ###
import FFN_Model_Cond_Dropout
import featureProcessing
import dataStatistics
import modelParameters as mp
mp = imp.reload(mp)


### Path to dataset ###
#dataPath = "/home/paperspace/Desktop/datasets/bcmRecordings/"
dataPath = "C:/Users/mhp/Documents/DNN_Datasets/bcmRecordings/"
#dataPath = "C:/Users/TobiasToft/Documents/dataset8_MultiTfNoise/"
feat_root_train = dataPath + "trainingInput/"
label_root_train = dataPath + "trainingReference/"

feat_root_val = dataPath + 'validationInput/'
label_root_val  = dataPath + 'validationReference/'


### Hyperparameters for tunig ###
hp_learning_rate    = [0.00001, 0.000001]
random.shuffle(hp_learning_rate)
hp_hidden_units     = [256, 512, 1024]#[256, 512, 1024, 2048]
random.shuffle(hp_hidden_units)
hp_nfft             = [128, 256, 512]
random.shuffle(hp_nfft)
hp_batch_size       = [32]
#random.shuffle(hp_batch_size)

### Removes old Tensorboard event files ###
for root, dirs, files in os.walk('./logs/', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

hp_tuning_results = np.empty((7))
hp_run = 1
movingAverageOrder = 3
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
                                                           500000, 0.96, staircase=True)
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


                ### Model variable initializer ###
                train_loss = []
                train_loss_mean = []
                val_loss = []
                val_loss_mean = []
                val_loss_moving = []
                val_loss_last_mean = 1000
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



                ### Data statistics ###
                print('Calculating data statistics ...')
                toc = time.time()
                featMean, featStd = dataStatistics.calcMeanAndStd(label_root_train,mp.DATASET_SIZE_TRAIN,nfft,mp.STFT_OVERLAP,mp.BIN_WIZE)
                toc = time.time()
                print(np.round(toc-tic,2),"secs to calc data stats")


                print('Hyperparameter tuning run: ' + str(hp_run))
                print('Rate: ' + str(rate) + ' Hidden Units: ' + str(units) + ' FFT Size: ' + str(nfft))

                print('Training...')

                with tf.Session() as sess:
                    if not os.path.exists('logs/run' + str(hp_run)):
                        os.mkdir('logs/run' + str(hp_run))
                        os.mkdir('logs/run' + str(hp_run) + '/train')
                        os.mkdir('logs/run' + str(hp_run) + '/val')
                        writer_train = tf.summary.FileWriter('logs/run' + str(hp_run) + '/train', sess.graph)
                        writer_val = tf.summary.FileWriter('logs/run' + str(hp_run) + '/val', sess.graph)
                        writer_val_smooth = tf.summary.FileWriter('logs/run' + str(hp_run) + '/val_smooth', sess.graph)

                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(max_to_keep=1)

                    firstRun = True
                    for epoch in range(1,mp.MAX_EPOCHS+1):
                        tic = time.time()
                        finalPreds = np.empty((0,numberClasses))

                        if trainingBool:
                            train_loss_mean = []

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

                                    fetches_train = [train_op,loss]
                                    feed_dict_train = {next_feat_pl: next_feat, next_label_pl: next_label, keepProb: mp.KEEP_PROB_TRAIN,is_train: True}

                                    # running the traning optimizer
                                    res_train = sess.run(fetches=fetches_train,
                                    feed_dict=feed_dict_train)

                                    train_loss.append(res_train[1])

                                    currentGlobalStep = sess.run(tf.train.get_global_step())
                                    ### Validate every x global step ###
                                    if np.mod(currentGlobalStep,mp.VALID_EVERY) == 0:
                                        #### Validation ####
                                        valFirstFile = True
                                        for file in allFilesVal:
                                            filePathFeat_val = feat_root_val + file
                                            filePathRef_val = label_root_val + file

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

                                            ### Validation flag ###
                                            val_loss_mean.append(np.mean(val_loss))
                                            val_loss = []

                                            valFirstFile = False

                                        ### Validation rutines ###
                                        val_loss_mean = np.mean(val_loss_mean)
                                        train_loss_mean.append(np.mean(train_loss))
                                        val_loss_moving.append(val_loss_mean)
                                        if len(val_loss_moving) > mp.MOVING_AVERAGE_ORDER:#*3:
                                            #val_loss_running = dsp.filtfilt(np.ones(mp.MOVING_AVERAGE_ORDER)/mp.MOVING_AVERAGE_ORDER,1,val_loss_moving,padtype='constant')
                                            #val_loss_last_mean = val_loss_running[-1]
                                            val_loss_last_mean = np.mean(val_loss_moving[-mp.MOVING_AVERAGE_ORDER:])
                                        else:
                                            val_loss_last_mean = np.mean(val_loss_moving)


                                        print('Number of global steps: %s' % currentGlobalStep)
                                        print("Training loss: ", np.mean(train_loss_mean), " Validation loss: ", val_loss_mean)

                                        ### Loss summary to Tensorboard ###
                                        if mp.DECAYING_LEARNING_RATE:
                                            rate_tb = np.float32(sess.run(learning_rate))
                                            summ = sess.run(tf_learning_rate_summary,feed_dict={rate_sum: rate_tb})
                                            writer_train.add_summary(summ, int(currentGlobalStep/mp.VALID_EVERY))
                                            writer_train.flush()


                                        summ1 = sess.run(tf_loss_summary,feed_dict={loss_sum: np.mean(train_loss_mean)})
                                        writer_train.add_summary(summ1, int(currentGlobalStep/mp.VALID_EVERY))
                                        writer_train.flush()

                                        summ2 = sess.run(tf_loss_summary,feed_dict={loss_sum: val_loss_mean})
                                        writer_val.add_summary(summ2, int(currentGlobalStep/mp.VALID_EVERY))
                                        writer_val.flush()

                                        summ3 = sess.run(tf_loss_summary,feed_dict={loss_sum: val_loss_last_mean})
                                        writer_val_smooth.add_summary(summ3, int(currentGlobalStep/mp.VALID_EVERY))
                                        writer_val_smooth.flush()


                                        ### Early stopping ###
                                        print('Best: ' + str(val_loss_best) + ' Current: ' + str(val_loss_mean) + ' Mean: ' + str(val_loss_last_mean))
                                        print()
                                        val_loss_mean = []
                                        if val_loss_last_mean < val_loss_best:
                                            val_loss_best = val_loss_last_mean

                                            trainingBool = True
                                            validationBool = True

                                            bestCount = 0

                                            # Save model
                                            saveStr = './savedModels_HP_Tune/hp_chechpoint' + str(rate) + '_' + str(units) + '_' + str(nfft) + '_' + str(batch) + '_' + str(epoch) + '.ckpt'
                                            saver.save(sess, saveStr)

                                        elif bestCount < mp.STOP_COUNT:
                                            trainingBool = True
                                            validationBool = True

                                            bestCount += 1
                                        else:
                                            trainingBool = False
                                            validationBool = False

                                # End of training loop
                                train_loss_mean.append(np.mean(train_loss))
                                train_loss = []


                            print('End of epoch:' + str(epoch) + " Number of global steps:" + str(sess.run(tf.train.get_global_step())))
                            toc = time.time()
                            print(np.round(toc-tic,2),"s")
                            print()


                        else:
                            break


                    #hp_tuning_results = np.concatenate((hp_tuning_results,np.asarray([rate,units,nfft,batch,val_loss_best])),axis=0)
                    hp_tuning_results = np.row_stack((hp_tuning_results,np.asarray([hp_run,rate,units,nfft,batch,val_loss_best,epoch])))
                    hp_run += 1

                    pandaArray = pd.DataFrame(hp_tuning_results)
                    pandaArray.columns = ['Run Number','Learning Rate','Hidden Units','FFT Size','Batch Size','Validation Loss','Number of epochs']
                    pandaArray = pandaArray.sort_values(by='Validation Loss')
                    resultsFilePath = 'hyperparameterTuningResults.xlsx'
                    pandaArray.to_excel(resultsFilePath,index=False)



writer_train.close()
writer_val.close()
writer_val_smooth.close()
print('Training done!')


pandaArray = pd.DataFrame(hp_tuning_results)
pandaArray.columns = ['Run Number','Learning Rate','Hidden Units','FFT Size','Batch Size','Validation Loss','Number of epochs']
pandaArray = pandaArray.sort_values(by='Validation Loss')
resultsFilePath = 'hyperparameterTuningResults.xlsx'
pandaArray.to_excel(resultsFilePath,index=False)
