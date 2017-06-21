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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

from keras.layers.recurrent import LSTM

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
from DNN_no_window_cross_validation import divide_training_data


def format_sequence_data(inputdim, outputdim, batchsize, x, y):
    yy = [0] * outputdim
    yy[-1] = 1
    while (x.shape[0] % batchsize != 0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    return x, y


def TrainBatch(layer, nodes, BATCH_SIZE, windowsize, portion):
    print('Loading data...')
    train_xxx_ori = np.loadtxt('trainvalidtest_x_windowing_'+ str(windowsize) + '.txt')
    train_yyy_ori = np.loadtxt('trainvalidtest_y_windowing_'+ str(windowsize) + '.txt')
    batch_size = 50
    INPUT_DIM = train_xxx_ori.shape[1]
    OUTPUT_DIM = train_yyy_ori.shape[1]
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer)+'layer'+str(nodes)+'BLSTM' + 'window_size' + str(windowsize) + 'training_data'+ str(portion) + 'batch_size' + str(batch_size)
    print('Loading data...')
    print('original train_xx shape:', train_xxx_ori.shape)
    print('original train_yy shape:', train_yyy_ori.shape)
    batch= BATCH_SIZE
    print('Build model...')
    cvscores = []
    cvscores_test = []
    cv_log = open('cv_log+' + MODEL_NAME + '.txt', 'w')
    for i in range(9):  # add test set to share another 10%, only validate 9 times!
        train_xxx, train_yyy, valid_xxx, valid_yyy, test_xxx, test_yyy = divide_training_data(10, portion, i, train_xxx_ori, train_yyy_ori)
        print('Shape for cross validation...')
        print('train_xx shape:', train_xxx.shape)
        print('train_yy shape:', train_yyy.shape)
        print('valid_xx shape:', valid_xxx.shape)
        print('valid_yy shape:', valid_yyy.shape)
        print('test_xx shape:', test_xxx.shape)
        print('test_yy shape:', test_yyy.shape)
        train_xxx, train_yyy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, batch_size, train_xxx, train_yyy)
        valid_xxx, valid_yyy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, batch_size, valid_xxx, valid_yyy)
        test_xxx, test_yyy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, batch_size, test_xxx, test_yyy)
        train_xxx = TwoToThree.TwoToThree(train_xxx, int(train_xxx.shape[0] / batch), int(batch), INPUT_DIM)
        train_yyy = TwoToThree.TwoToThree(train_yyy, int(train_yyy.shape[0] / batch), int(batch), OUTPUT_DIM)
        valid_xxx = TwoToThree.TwoToThree(valid_xxx, int(valid_xxx.shape[0] / batch), int(batch), INPUT_DIM)
        valid_yyy = TwoToThree.TwoToThree(valid_yyy, int(valid_yyy.shape[0] / batch), int(batch), OUTPUT_DIM)
        test_xxx = TwoToThree.TwoToThree(test_xxx, int(test_xxx.shape[0] / batch), int(batch), INPUT_DIM)
        test_yyy = TwoToThree.TwoToThree(test_yyy, int(test_yyy.shape[0] / batch), int(batch), OUTPUT_DIM)
        print('train_xx shape:', train_xxx.shape)
        print('train_yy shape:', train_yyy.shape)
        print('valid_xx shape:', valid_xxx.shape)
        print('valid_yy shape:', valid_yyy.shape)
        print('test_xx shape:', test_xxx.shape)
        print('test_yy shape:', test_yyy.shape)
        model = Sequential()
        #model.add(Embedding(36, 256, input_length=batch))
        #model.add(Dense(output_dim=256, init='glorot_uniform', activation='tanh', input_dim= 36))
        #model.add(LSTM(output_dim=48, init='glorot_uniform', inner_init='orthogonal', activation='softmax', inner_activation='tanh'))  # try using a GRU instead, for fun
        model.add(Bidirectional(LSTM(return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_dim=INPUT_DIM, units=HIDDEN_NODE, kernel_initializer="glorot_uniform"),input_shape=train_xxx.shape))
        for i in range(layer-1):
            model.add(Bidirectional(LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(OUTPUT_DIM))
        model.add(Activation('softmax')) # need time distributed softmax??
        #model.add(Dropout(0.2))
        # try using different optimizers and different optimizer configs
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5) # set up early stopping
        print("Train...")
        checkpointer = ModelCheckpoint(filepath=MODEL_NAME+".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
        hist = model.fit(train_xxx, train_yyy, batch_size=batch_size, epochs=200, validation_data=(valid_xxx,valid_yyy), shuffle=True, verbose=1, callbacks=[checkpointer, early_stopping])

        #SaveModelLog.Save(MODEL_NAME, hist, model, test_xxx, test_yyy)
        scores = model.evaluate(valid_xxx, valid_yyy, verbose=0)
        scores_test = model.evaluate(test_xxx, test_yyy, verbose=0)
        print(' valid_acc: ', scores[1])
        cvscores.append(scores[1] * 100)
        cvscores_test.append(scores_test[1] * 100)
        # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)
    print(np.mean(cvscores), np.std(cvscores))
    print(MODEL_NAME, file=cv_log)
    print('valid:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
    for i in range(len(cvscores_test)):
        print('Test:', i, cvscores_test[i], '%', file=cv_log)
    print('Test:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)


