#!/usr/bin/env python
#The MIT License
#
#Copyright (c) 2017 Geoffrey So

import librosa
import numpy as np
import librosa.display
from scipy.fftpack import dct
import time
import os
import pickle
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# decorator function to time different function calls
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result

    return timed

def plot_melspec(S, hop_length):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=fmax, x_axis='time', hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

# extract 2D features from audio files
PROJPATH = r'C:\Users\gso\Documents\PythonProjects\music_genre'
os.chdir(PROJPATH)
GENRES = PROJPATH + '\genres'
GENRESPATH = glob.glob(GENRES+'\*')
MELSPECPATH = PROJPATH + '\MelSpec'
createpath(MELSPECPATH)
MFCCPATH = PROJPATH + '\MFCC'
createpath(MFCCPATH)
PSTCPATH = PROJPATH + '\PSTC'
createpath(PSTCPATH)
STACKEDPATH = PROJPATH + '\StackedInfo'
createpath(STACKEDPATH)

n_mels = 128
fmax = 8000
hop_length = 2048
# add names to columns list for pandas dataframe
columns = []
nameTarget = []
# shave 1 sec from all audio, since they are all 29.X approx 30 sec, need exact
maxtimelength = 29
timeresolution = int(np.ceil(22050*maxtimelength/hop_length)) # should be sr*maxtimelength/hop_length

GENDATA = False
VERBOSE = True
if GENDATA:
    if os.path.exists('nameTarget.csv'):
        os.remove('nameTarget.csv')
    for i, genre in enumerate(GENRESPATH):
        files = glob.glob(genre+'\*.au')
        genreName = genre.split('\\')[-1]
        columns.append(genreName)
        if VERBOSE:
            print('processing %s' % genreName)
        for file in files:
            y, sr = librosa.load(file)
            y = y[:(sr * maxtimelength)]
            npyName = ''.join(file.split('\\')[-1].split('.')[:2])+'.npy'

            # Mel Frequency Spectrogram
            MelSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
            if len(MelSpec[0])!= timeresolution:
                print('TIME RESOLUTION PROBLEM')
                break
            MelSpec.tofile(MELSPECPATH + '\\' + npyName)

            # MFCC
            MFCC = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels, hop_length=hop_length)
            MFCC.tofile(MFCCPATH + '\\' + npyName)

            # PSTC
            PSTC = dct(dct(MelSpec.transpose()).transpose())
            PSTC.tofile(PSTCPATH + '\\' + npyName)

            # stacking information into one image
            stackedinfo = np.vstack((MelSpec, MFCC, PSTC))
            stackedinfo.tofile(STACKEDPATH + '\\' + npyName)

            # adding the file name and target to list as a tuple
            nameTarget.append((npyName, genreName))

    pd.DataFrame(nameTarget, columns=['name','class']).to_csv('nameTarget.csv', index=False)
else:
    print('skipping data generation')

'''
# CNN from tflearn convnet_highway_mnist.py

X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
'''

df = pd.read_csv('nameTarget.csv')
files = df['name']
classes = df['class']
target = pd.get_dummies(classes).values
x_dim = timeresolution

#SOURCEPATH = MELSPECPATH
#y_dim = n_mels

SOURCEPATH = MFCCPATH
y_dim = n_mels

#SOURCEPATH = PSTCPATH
#y_dim = n_mels

#SOURCEPATH = STACKEDPATH
#y_dim = n_mels*3

trainX = []
testX = []
trainY = []
testY = []
randomseed = 9999

# using last 10% as test
for i in range(1,len(files)+1):
    # load using 'np.fromfile' for ndarray, 'np.load' for array
    if VERBOSE:
        print('start processing %s' % files[i - 1])
    if i % 10:
        trainX.append(np.fromfile(SOURCEPATH + '\\' + files[i - 1]).reshape([y_dim, x_dim, 1]))
        trainY.append(target[i - 1])
    else:
        testX.append(np.fromfile(SOURCEPATH + '\\' + files[i - 1]).reshape([y_dim, x_dim, 1]))
        testY.append(target[i - 1])

#trainX, validX, trainY, validY = train_test_split(trainX, trainY, train_size = 800, random_state = randomseed)

trainX = np.array(trainX)
#trainX = trainX.reshape([-1, y_dim, x_dim, 1])
trainY = np.array(trainY)

# Building convolutional network
network = input_data(shape=[None, y_dim, x_dim, 1], name='input')
network = conv_2d(network, 16, 3, strides=1, activation='relu', regularizer="L2")
#network = max_pool_2d(network, 2)
#network = dropout(network, 0.9)
network = local_response_normalization(network)
network = conv_2d(network, 32, 5, strides=2, activation='relu', regularizer="L2")
#network = max_pool_2d(network, 2)
#network = dropout(network, 0.9)
network = local_response_normalization(network)
network = conv_2d(network, 64, 10, strides=4, activation='relu', regularizer="L2")
#network = max_pool_2d(network, 2)
#network = dropout(network, 0.9)
network = local_response_normalization(network)
network = conv_2d(network, 128, 15, strides=6, activation='relu', regularizer="L2")
#network = max_pool_2d(network, 2)
#network = dropout(network, 0.9)
network = local_response_normalization(network)

network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.9)

network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.0003,
                     loss='categorical_crossentropy', name='target')

# Training

model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='tflearn_logs', checkpoint_path='./model_checkpoints/my_model')
#model.fit(trainX, trainY, n_epoch=20, validation_set=(validX, validY), show_metric=True, run_id='music_genre')
model.fit(trainX, trainY, n_epoch=20, validation_set=(testX, testY), show_metric=True, run_id='music_genre')
'''
predY = model.predict(testX)
acc = accuracy_score(testY, predY)
print('Final Accuracy %f' % acc)
'''
