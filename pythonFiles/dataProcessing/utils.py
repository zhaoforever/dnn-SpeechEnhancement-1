import tensorflow as tf
from scipy.io import wavfile
import numpy as np
import os
from scipy import signal

def qualityAssessment(predictionPath,inputPath,referencePath):
	import pesq_calc
	import MCC_calc
	### Dataset and feature extraction parameters ###
	NFFT = 512

	# initialize empty arrays
	MCC_all_pred = []
	MCC_all_input = []
	segMCC_all_pred = np.empty((int(NFFT/2+1),0), int)
	segMCC_all_input = np.empty((int(NFFT/2+1),0), int)
	pesqScore_all_pred = []
	pesqScore_all_input = []

	allFiles = os.listdir(predictionPath)

	n = 1
	for file in allFiles:
		#tic = time.time()
		filePathPrediction = predictionPath + file
		filePathInput = inputPath + file.replace('pred','feat')
		filePathReference = referencePath + file.replace('pred','ref')

		### MCC ###

		# Prediction
		MCC_pred,segMCC_pred,freq = MCC_calc.MCC_calc(filePathReference,filePathPrediction,NFFT)
		MCC_all_pred.append(MCC_pred)
		segMCC_all_pred = np.append(segMCC_all_pred,segMCC_pred,axis=1)
		# Input
		MCC_input,segMCC_input,freq = MCC_calc.MCC_calc(filePathReference,filePathInput,NFFT)
		MCC_all_input.append(MCC_input)
		segMCC_all_input = np.append(segMCC_all_input,segMCC_input,axis=1)



		pesqScore_pred = pesq_calc.pesq_calc(filePathReference,filePathPrediction)
		pesqScore_all_pred = np.append(pesqScore_all_pred,pesqScore_pred)

		pesqScore_input = pesq_calc.pesq_calc(filePathReference,filePathInput)
		pesqScore_all_input = np.append(pesqScore_all_input,pesqScore_input)

		print("File nr. " +  str(n) + " done!")
		n+=1
		#toc = time.time()
		#print(toc-tic)

	MCC_mean_pred = np.mean(MCC_all_pred)
	MCC_mean_input = np.mean(MCC_all_input)
	segMCC_mean_pred = np.mean(segMCC_all_pred,axis=1)
	segMCC_mean_input = np.mean(segMCC_all_input,axis=1)

	pesqScore_mean_pred = np.mean(pesqScore_all_pred)
	pesqScore_mean_input = np.mean(pesqScore_all_input)


	#print("Size of dataset: " + str(np.size(allFiles)) + " wav files" + "\n")
	#print("            MCC   PESQ")
	#print("Input:      " + str(np.round(MCC_mean_input,2)) + " " + str(np.round(pesqScore_mean_input,2)))
	#print("Prediction: " + str(np.round(MCC_mean_pred,2)) + " " + str(np.round(pesqScore_mean_pred,2)))
	#print("Difference: " + str(np.round((MCC_mean_pred-MCC_mean_input),2)) + " " + str(np.round((pesqScore_mean_pred-pesqScore_mean_input),2)))

	return MCC_mean_input, pesqScore_mean_input, MCC_mean_pred, pesqScore_mean_pred, segMCC_mean_input, segMCC_mean_pred, freq

def freezeModel(savedModelPath):

	saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(savedModelPath) + '.meta', clear_devices=True)
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()
	sess = tf.Session()
	saver.restore(sess, tf.train.latest_checkpoint(savedModelPath))
	# Output from the model that needs to be freezed
	output_node_names = "out/BiasAdd"
	output_graph_def = tf.graph_util.convert_variables_to_constants(sess,input_graph_def,output_node_names.split(","))
	output_graph = savedModelPath + "myFrozenModel.pb"
	with tf.gfile.GFile(output_graph, "wb") as f:
	    f.write(output_graph_def.SerializeToString())
	sess.close()
	return

def adjustSNR(x,dB):
	x = x/np.max(np.abs(x))
	xRms = np.sqrt(np.mean(np.square(x)))
	x = x*10**((dB-20*np.log10(xRms/2e-5))/20);
	return x

def muLawCompression(wavSamples):
	mu = 255

	xMin = np.min(wavSamples)
	xMax = np.max(wavSamples)

	delta = (xMax-xMin)/mu
	return np.floor( (wavSamples-xMin)/delta)

def hypTanCompression(data):
	# Hyperbolic tangent compression
	Q = 1.0
	C = 0.5
	return Q * ((1.0-np.exp(-C*data))/(1.0+np.exp(-C*data)))

def hypTanCompression_inv(data):
	# Inverte Hyperbolic tangent compression
	Q = 1.0
	C = 0.5
	return (-1.0/C)*(np.log((Q - data)/(Q + data)))


def wavToSamples(wav_file):
	fs, wav_data = wavfile.read(wav_file)
	assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
	wav_data = wav_data / 32767.0  # Convert to [-1.0, +1.0]

	return wav_data, fs

def STFT(x,fs,wL,nOverlap):
	if signal.check_COLA('hann',wL,nOverlap):
		f, t, Zxx = signal.stft(x, fs, nperseg=wL,noverlap=nOverlap,return_onesided=True)
	else:
		print('COLA constrain not met! Change STFT parameters!')
		return
	#Zxx_abs = np.abs(Zxx)
	#Zxx_phi = np.arctan2(Zxx.imag,Zxx.real)

	return np.abs(Zxx), np.arctan2(Zxx.imag,Zxx.real), t, f

def ISTFT(X_abs,X_phi,fs,nfft,nOverlap):

	X_re = X_abs*np.cos(X_phi)
	X_im = X_abs*np.sin(X_phi)

	X_complex = X_re+1j*X_im

	_, xrec = signal.istft(X_complex, fs, nperseg=nfft,noverlap=nOverlap)

	return xrec


def hz2mel(freq):
    # Convert Hertz to Mels - numpy array, or single value
    return 2595 * np.log10(1+freq/700.)

def mel2hz(mel):
    # Convert Mels to Hertz - numpy array, or single value
    return 700*(10**(mel/2595.0)-1)

def freqMelConversionMatrix(linFreqBins,melFreqBins,freq_low_hz,freq_up_hz,fs):
	#  Frequency scale conversion matrix
	#   LinFrqBins   : Number of input frequency bins
	#   LogFrqBins   : Number of output frequency bins
	#   Fmin         : Lowest output bin [Hz]
	#   Fmax         : Highest output bin [Hz]
	#   Fs           : Sample rate [Hz]
	freq_low_mel = utils.hz2mel(freq_low_hz)
	freq_up_mel = utils.hz2mel(freq_up_hz)

	freq_mel = np.linspace(freq_low_mel, freq_up_mel, num=melFreqBins+2)

	freq_hz = utils.mel2hz(freq_mel)

	f_bins = np.floor((linFreqBins+1)*freq_hz/fs) # round frequencies to fft-bin

	fbank = np.zeros([melFreqBins,linFreqBins//2+1])
	print(fbank.shape)

	for j in range(0,melFreqBins):
	    for i in range(int(f_bins[j]), int(f_bins[j+1])):
	        fbank[j,i] = (i - f_bins[j]) / (f_bins[j+1]-f_bins[j])
	    for i in range(int(f_bins[j+1]), int(f_bins[j+2])):
	        fbank[j,i] = (f_bins[j+2]-i) / (f_bins[j+2]-f_bins[j+1])

	return fbank


def cutSignal(wav_data,fs,seg):
	segLength = fs*seg # segment length in samples
	T0 = len(wav_data)/fs # length of wav_file in seconfs

	segments = [] # Initializing vector

	# If length of segment does not match length of wav file, the wav file is zero padded
	if (T0/seg) % 1 != 0:

		segN = int(len(wav_data)/segLength)*fs
		segPad = np.abs(segLength-(len(wav_data)-segN))

		wav_data = np.concatenate((wav_data,np.zeros(segPad)),axis=0)

	# segmentaion of the wav file
	T0 = len(wav_data)/fs
	for i in range(0,int(T0/seg)):
		segData = wav_data[i*segLength:i*segLength+segLength]

		segments.append(segData)

	return segments


def _wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tfrecordWrite(tfrecords_filename,data_input,data_label):
# Construction of TFRecords:
# TFRecords is the binary file-format used internally in TensorFlow which allows
# for high-performance reading and processing of datasets.
	with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:

		# restricting memory usage:
	    config = tf.ConfigProto()
	    config.gpu_options.allocator_type = 'BFC'
	    config.gpu_options.per_process_gpu_memory_fraction = 0.90

	    # convert into proper data type:
	    data_input_raw = data_input.tostring()
	    data_label_raw = data_label.tostring()

	    # dict:
	    data = \
	        {
	        'data_input': _wrap_bytes(data_input_raw),
	        'data_label': _wrap_bytes(data_label_raw),
	        'data_shape_dim1': _wrap_int64(int(data_input.shape[0]))
	        }

	    # Wrap the data as TensorFlow Features.
	    feature = tf.train.Features(feature=data)

	    # Wrap again as a TensorFlow Example.
	    example = tf.train.Example(features=feature)

	    # Write the serialized data to the TFRecords file.
	    writer.write(example.SerializeToString())


def tfrecordRead(tfrecords_filename):
	# TODO: tfRecords - Compare tfrecords with NPY files, Speed performance?
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

	data_input_tot = []
	data_label_tot = []

	for string_record in record_iterator:

	    example = tf.train.Example()
	    example.ParseFromString(string_record)

	    data_input = (example.features.feature['data_input']
	                                 .bytes_list
	                                 .value[0])

	    data_label = (example.features.feature['data_label']
	                                  .bytes_list
	                                  .value[0])
	    data_shape_dim1 = (example.features.feature['data_shape_dim1']
	                                  .int64_list
	                                  .value[0])

	    data_input = np.fromstring(data_input, dtype=np.float64)
	    data_label = np.fromstring(data_label, dtype=np.float64)

	    data_input_tot.append(data_input)
	    data_label_tot.append(data_label)


	data_input_tot = np.asarray(data_input_tot)
	data_label_tot = np.asarray(data_label_tot)
	data_input_tot = data_input_tot.reshape((data_shape_dim1,-1))
	data_label_tot = data_label_tot.reshape((data_shape_dim1,-1))

	return data_input_tot, data_label_tot


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def dispBound(data):

	print("Min value: ",np.min(data))
	print("Mean value: ",np.mean(data))
	print("Max value: ",np.max(data))
	return

def featureExtractMag512(x,fs):


	nfft = 512
	X_abs,X_phi,_,_ = STFT(x,fs,nfft,int(nfft*0.75))

	# Feature normalization
	datasetPath = "C:/Users/s123028/dataset7_multiBcmTf/"
	trainingStats = np.load(datasetPath + "trainingStats.npy")

	featMean = trainingStats[0]
	featStd = trainingStats[1]

	X_abs = np.log10(X_abs + 1e-7)
	featMean = np.mean(X_abs)
	X_abs = X_abs - featMean
	featStd = np.std(X_abs)
	X_abs = X_abs/featStd

	X_abs = np.transpose(X_abs)

	X_abs = np.float32(X_abs)

	return X_abs, X_phi

def featureReconstructMag512(xPreds,X_phi,fs):

	datasetPath = "C:/Users/s123028/dataset7_multiBcmTf/"
	trainingStats = np.load(datasetPath + "trainingStats.npy")

	featMean = trainingStats[0]
	featStd = trainingStats[1]

	finalPreds = xPreds

	finalPreds1 = finalPreds * featStd
	finalPreds2 = finalPreds1 + featMean
	finalPreds3 = 10**(finalPreds2)-1e-7
	y = ISTFT(finalPreds3,X_phi,fs,512,int(512*0.75))
	y = y/np.max(np.abs(y))

	return y
