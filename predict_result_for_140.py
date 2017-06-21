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
#from keras.utils.visualize_util import plot # draw fig
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from scipy.io import loadmat
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import h5py
import os
import numpy as np
import keras.callbacks as CB
import sys
import string
import time
import SaveModelLog
from get_input_and_output import get_chord_list, get_chord_line, calculate_freq
from music21 import *
from DNN_no_window_cross_validation import divide_training_data
def get_predict_file_name():
    string = 'test'
    cwd = '.\\bach_chorales_scores\\transposed_MIDI\\'
    filename = []
    num_salami_slices = []
    for fn in os.listdir(cwd):
        #print(fn)
        if fn[-3:] == 'mid':
            if (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop','r')

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            length = len(sChords.notes)
            filename.append(fn[0:3])
            num_salami_slices.append(length)
    return filename, num_salami_slices

def train_and_predict(layer,nodes,windowsize,portion):
    print('Loading data...')
    #log=open('256LSTM256LSTMNN48lr=0.1dp=0.5.txt','w+')
    #AllData='musicALL7.mat'
    train_xx = np.loadtxt('traintraintrain_x_windowing_10-70.txt')
    train_yy = np.loadtxt('traintraintrain_y_windowing_10-70.txt')
    valid_xx = np.loadtxt('validvalidvalid_x_windowing_10-70.txt')
    valid_yy = np.loadtxt('validvalidvalid_y_windowing_10-70.txt')
    test_xx = np.loadtxt('testtesttest_x_windowing_10-70.txt')
    test_yy = np.loadtxt('testtesttest_y_windowing_10-70.txt')
    #train_xxx = train_xxx_ori[:portion*train_xxx_ori.shape[0]]
    #train_yyy = train_yyy_ori[:portion * train_yyy_ori.shape[0]]
    #all_xx = np.loadtxt('all_x.txt')
    #all_yy = np.loadtxt('all_y.txt')
    #np.random.shuffle(all_xx)
    #np.random.shuffle(all_yy)
    '''train_xx = all_xx[:int(len(all_xx)*0.8)]
    valid_xx = all_xx[int(len(all_xx)*0.8):int(len(all_xx)*0.9)]
    test_xx = all_xx[int(len(all_xx)*0.9):]
    train_yy = all_yy[:int(len(all_yy)*0.8)]
    valid_yy = all_yy[int(len(all_yy)*0.8):int(len(all_yy)*0.9)]
    test_yy = all_yy[int(len(all_yy)*0.9):]'''
    #max_features = 20000
    #maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 50
    INPUT_DIM = train_xx.shape[1]
    OUTPUT_DIM = train_yy.shape[1]
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer)+'layer'+str(nodes)+'DNN' + 'window_size' + str(windowsize) + 'training_data'+ str(portion)
    print('Loading data...')
    print('original train_xx shape:', train_xx.shape)
    print('original train_yy shape:', train_yy.shape)
    print('valid_xx shape:', valid_xx.shape)
    print('valid_yy shape:', valid_yy.shape)
    print('test_xx shape:', test_xx.shape)
    print('test_yy shape:', test_yy.shape)
    print('Build model...')
    #cvscores = []
    #cvscores_test = []
    #cv_log = open('cv_log+' + MODEL_NAME + '.txt', 'w')
    #for i in range(9):  # add test set to share another 10%, only validate 9 times!
        #train_xx, train_yy, valid_xx, valid_yy, test_xx, test_yy = divide_training_data(10, portion, i, train_xxx_ori, train_yyy_ori)
    print('Shape for cross validation...')
    print('train_xx shape:', train_xx.shape)
    print('train_yy shape:', train_yy.shape)
    print('valid_xx shape:', valid_xx.shape)
    print('valid_yy shape:', valid_yy.shape)
    print('test_xx shape:', test_xx.shape)
    print('test_yy shape:', test_yy.shape)
    model = Sequential()
    # model.add(Embedding(36, 256, input_length=batch))
    model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh', input_dim=INPUT_DIM))
    model.add(Dropout(0.2))
    for i in range(layer - 1):
        # model.add(LSTM(output_dim=48, init='glorot_uniform', inner_init='orthogonal', activation='softmax', inner_activation='tanh'))  # try using a GRU instead, for fun
        # model.add(LSTM(input_dim=INPUT_DIM, output_dim=500, return_sequences=True, init='glorot_uniform'))
        # model.add(LSTM(output_dim=500, return_sequences=True))
        # model.add(LSTM(output_dim=500, return_sequences=True))
        # model.add(LSTM(48))
        model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh', input_dim=INPUT_DIM))
        model.add(Dropout(0.2))
    model.add(Dense(OUTPUT_DIM, init='uniform'))
    # model.add(Dropout(0.5)) # dropout does not add at output layer!!
    model.add(Activation('softmax'))  # need time distributed softmax??

    # try using different optimizers and different optimizer configs
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.1, decay=0.002, momentum=0.5,
              nesterov=False)  # lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # set up early stopping
    print("Train...")
    checkpointer = ModelCheckpoint(filepath=MODEL_NAME + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
    # hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=3,validation_split=0.2, shuffle=True, verbose=1, show_accuracy=True, callbacks=[early_stopping])
    hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=100, shuffle=True, verbose=1,
                     validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer])  # for debug

    # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)

    # visualize the result and put into file
    model = load_model(MODEL_NAME + ".hdf5")
    predict_y = model.predict(test_xx, verbose=0)
    score = model.evaluate(test_xx, test_yy, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    list_of_chords = get_chord_list(predict_y.shape[1], '0')
    list_of_chords.append('other')
    fileName, numSalamiSlices = get_predict_file_name()
    sum = 0
    for i in range(len(numSalamiSlices)):
        sum += numSalamiSlices[i]
    #input(sum)
    #input(predict_y.shape[0])


    length = len(fileName)
    a_counter = 0
    a_counter_correct = 0
    for i in range(length):
        f = open('predicted_result_' + fileName[i] + '.txt', 'w')
        num_salami_slice = numSalamiSlices[i]
        correct_num = 0
        for j in range(num_salami_slice):
            pointer = np.argmax(predict_y[a_counter])
            pointer_gt = np.argmax(test_yy[a_counter])
            if(pointer == pointer_gt):  # the label is correct
                correct_num += 1


            currentchord = list_of_chords[pointer]
            if((j+1)%10==0):
                print(currentchord, end='\n', file=f)
            else:
                print(currentchord, end=' ', file=f)
            a_counter += 1
        a_counter_correct += correct_num
        print(end='\n', file=f)
        #print('accucary: ' + str(correct_num/num_salami_slice), end='\n', file=f)
        #print('num of correct answers: ' + str(correct_num) + ' number of salami slices: ' + str(num_salami_slice), file=f)
        #print('accumulative accucary: ' + str(a_counter_correct / a_counter), end='\n', file=f)
        f.close()

    #np.savetxt('predict_y_windowing_1' + '.txt', predict_yy, fmt='%.1e')'''

if __name__ == "__main__":
    train_and_predict(2, 200, 1, 1)