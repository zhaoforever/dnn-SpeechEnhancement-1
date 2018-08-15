## Unfreeze model
import tensorflow as tf
import utils
import FFNModel
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
import sounddevice as sd
from scipy import signal
slim = tf.contrib.slim
from numpy import seterr,isneginf,isinf,isnan
from scipy.io import wavfile

datasetPath = "C:/Users/s123028/dataset1khz/"
trainingStats = np.load(datasetPath + "trainingStats.npy")

featMean = trainingStats[0]
featStd = trainingStats[1]

#filePath = "C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/thinLinc/Mag/CNN/wavPredictionFiles/featTestMultBCM.wav"
#filePath2 = "C:/Users/Mikkel/ABE_Master_thesis/PythonFiles/thinLinc/Mag/CNN/wavPredictionFiles/refTestMultBCM.wav"
filePath = "C:/Users/s123028/dataset1khz/TIMIT_val_feat/748_val_feat.wav"
filePath2 = "C:/Users/s123028/dataset1khz/TIMIT_val_ref/748_val_ref.wav"
#filePath = "C:/Users/s123028/dataset4/TIMIT_val_feat/999_feat.wav"
#filePath2 = "C:/Users/s123028/dataset4/TIMIT_val_ref/999_ref.wav"
#filePath = "C:/Users/s123028/dens_01_3k.wav"
#filePath2 = "C:/Users/s123028/dens_01_ref.wav"

#filePath2 = "C:/Users/Mikkel/ABE_Master_thesis_1/MatlabFiles/recordings/andw_03_ref.wav"

fs, x = wavfile.read(filePath)

#x, fs = utils.wavToSamples(filePath)
x = x[0:int(fs*6)]
#x = signal.decimate(x,3)
#fs = 16000
#x = x/np.max(np.abs(x))

fs2, x_ref = wavfile.read(filePath2)

#x_ref, fs2 = utils.wavToSamples(filePath2)
x_ref = x_ref[0:int(fs2*6)]
#x_ref = signal.decimate(x_ref,3)
#fs2 = 16000
#x_ref = x_ref/np.max(np.abs(x_ref))

wL = 512

features,X_phi,t,f= utils.STFT(x,fs,wL,int(wL*0.75))
features.shape
X_phi.shape

labels,L_phi,t2,f2= utils.STFT(x_ref,fs2,wL,int(wL*0.75))
labels.shape

#### MODEL ####
features = np.float32(features)
labels = np.float32(labels)

#cond1 = np.std(features,axis=1) > 0
#features = features[cond1,:]

#cond2 = features == 0
#features[cond2] += np.finfo(float).eps

features = np.log10(features + 1e-7)
print(np.isfinite(features).all())
features = features - featMean
features = features / featStd
print(np.isfinite(features).all())

featuresOrg = np.transpose(features)
#features = featuresOrg
features = featuresOrg[:,:]
features.shape

#r,c = features.shape
#features = np.ones(r,c)
#features = np.float32(features)




labels = np.log10(labels + 1e-7)
print(np.isfinite(labels).all())
labels = labels - featMean
labels = labels / featStd
print(np.isfinite(labels).all())

labels = np.transpose(labels)
labels.shape

#plt.pcolormesh(t, f, np.transpose(features))
#plt.title('STFT Magnitude')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

features_placeholder = tf.placeholder(features.dtype, features.shape)

dataFeatures = tf.data.Dataset.from_tensor_slices(features_placeholder)

testDataset = tf.data.Dataset.zip(dataFeatures)

testDataset = testDataset.batch(1)
testIterator = testDataset.make_initializable_iterator()
next_testFeat = testIterator.get_next()

next_testElement = next_testFeat

next_feat_pl = tf.placeholder(next_testFeat.dtype,next_testFeat.shape)
keepProb = tf.placeholder(tf.float32)

NUM_CLASSES = int(np.size(labels,axis=1))
print(NUM_CLASSES)

epoch_num = 1

finalPreds = []

### UNFREEZE MODEL ###

frozen_graph="./savedModels/myFrozenModel.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )


for op in graph.get_operations():
    print(op.name)

preds = graph.get_tensor_by_name("out/BiasAdd:0")
next_feat_pl = graph.get_tensor_by_name("inputFeatures:0")
keepProb = graph.get_tensor_by_name("keepProb:0")

sessInt = tf.Session()

with tf.Session(graph=graph) as sess:
    for epoch in range(0,epoch_num):
        sessInt.run(testIterator.initializer,
        feed_dict={features_placeholder: features})

        while True:
            try:
                next_feat = sessInt.run(next_testElement)


                fetches_test = [preds]
                feed_dict_test = {next_feat_pl: next_feat,keepProb:1.0}

                # running the traning optimizer
                res_test = sess.run(fetches=fetches_test, feed_dict=feed_dict_test)


                finalPreds.append(res_test[0])

            except tf.errors.OutOfRangeError:
                #print('End of epoch')
                #plt.imshow(res_test[0])
                break
finalPreds0 = np.transpose(np.asarray(finalPreds)[:,-1,:])
print('Training done!')



plt.pcolormesh(finalPreds0)
#plt.ylim((0, 1000))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
