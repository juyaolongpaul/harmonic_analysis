'''Train a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from scipy.io import loadmat
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import h5py
import os
import numpy as np
import keras.callbacks as CB
import sys
import string
import time
import TwoToThree
import SaveModelLog
from hcf import hcf
def TrainBatch(layer,nodes,BATCH_SIZE):

    print('Loading data...')

    train_xx = np.loadtxt('train_x.txt')
    train_yy = np.loadtxt('train_y.txt')
    valid_xx = np.loadtxt('valid_x.txt')
    valid_yy = np.loadtxt('valid_y.txt')
    test_xx = np.loadtxt('test_x.txt')
    test_yy = np.loadtxt('test_y.txt')
    print('train_xx shape:', train_xx.shape)
    print('train_yy shape:', train_yy.shape)
    print('valid_xx shape:', valid_xx.shape)
    print('valid_yy shape:', valid_yy.shape)
    print('test_xx shape:', test_xx.shape)
    print('test_yy shape:', test_yy.shape)
    #max_features = 20000
    #maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 16
    INPUT_DIM = train_xx.shape[1]
    OUTPUT_DIM = train_yy.shape[1]
    HIDDEN_NODE = nodes

    MODEL_NAME = str(layer)+'layer'+str(nodes)+'batch_size'+str(BATCH_SIZE)+'HiddenNodeLSTMDropout0.5ADAM'
    print('Loading data...')
    #(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                          #test_split=0.2)
    #print(len(train_xx), 'train sequences')
    #print(len(train_yy), 'test sequences')

    #print("Pad sequences (samples x time)")
    #train_xx = sequence.pad_sequences(train_xx, maxlen=maxlen)
    #X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    #print('X_test shape:', X_test.shape)
    batch= BATCH_SIZE
    if(train_xx.shape[0]/batch % 1 != 0 or valid_xx.shape[0]/batch % 1 != 0 ):
        batch = hcf(train_xx.shape[0], valid_xx.shape[0])
        print("最大公约数：" + str(batch))
    train_xxx=TwoToThree.TwoToThree(train_xx,int(train_xx.shape[0]/batch),int(batch),INPUT_DIM)
    train_yyy=TwoToThree.TwoToThree(train_yy,int(train_yy.shape[0]/batch),int(batch),OUTPUT_DIM)
    valid_xxx=TwoToThree.TwoToThree(valid_xx,int(valid_xx.shape[0]/batch),int(batch),INPUT_DIM)
    valid_yyy=TwoToThree.TwoToThree(valid_yy,int(valid_yy.shape[0]/batch),int(batch),OUTPUT_DIM)
    test_xxx=TwoToThree.TwoToThree(test_xx,int(test_xx.shape[0]/batch),int(batch),INPUT_DIM)
    test_yyy=TwoToThree.TwoToThree(test_yy,int(test_yy.shape[0]/batch),int(batch),OUTPUT_DIM)
    #print ('pause')


    model = Sequential()
    #model.add(Embedding(36, 256, input_length=batch))
    #model.add(Dense(output_dim=256, init='glorot_uniform', activation='tanh', input_dim= 36))
    #model.add(LSTM(output_dim=48, init='glorot_uniform', inner_init='orthogonal', activation='softmax', inner_activation='tanh'))  # try using a GRU instead, for fun
    model.add(LSTM(input_dim=INPUT_DIM, output_dim=HIDDEN_NODE, return_sequences=True, init='glorot_uniform'))
    model.add(Dropout(0.2))
    for i in range(layer-1):
        model.add(LSTM(output_dim=HIDDEN_NODE, return_sequences=True))
        model.add(Dropout(0.2))
    model.add(Dense(OUTPUT_DIM))
    model.add(Activation('softmax')) # need time distributed softmax??
    #model.add(Dropout(0.2))
    # try using different optimizers and different optimizer configs
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2) # set up early stopping
    print("Train...")
    checkpointer = ModelCheckpoint(filepath=MODEL_NAME+".hdf5", verbose=1, save_best_only=True, monitor='val_acc')
    hist = model.fit(train_xxx, train_yyy, batch_size=batch_size, nb_epoch=200, validation_data=(valid_xxx,valid_yyy), shuffle=True, verbose=1, callbacks=[checkpointer, early_stopping])

    SaveModelLog.Save(MODEL_NAME, hist, model, test_xxx, test_yyy)
    score = model.evaluate(test_xxx, test_yyy, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


