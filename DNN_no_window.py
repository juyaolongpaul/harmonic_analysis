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
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.utils import np_utils
#from keras.utils.visualize_util import plot # draw fig
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Bidirectional, RNN, SimpleRNN, TimeDistributed
from keras.datasets import imdb
from scipy.io import loadmat
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import keras.backend as K
import keras.callbacks as CB
import sys
import string
import time
import SaveModelLog
from get_input_and_output import get_chord_list, get_chord_line, calculate_freq
from music21 import *
from DNN_no_window_cross_validation import divide_training_data
import TwoToThree

def format_sequence_data(inputdim, outputdim, batchsize, x, y):
    yy = [0] * outputdim
    yy[-1] = 1
    while (x.shape[0] % batchsize != 0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    return x, y


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

def FineTuneDNN(layer,nodes,windowsize,portion):
    print('Loading data...')
    #log=open('256LSTM256LSTMNN48lr=0.1dp=0.5.txt','w+')
    #AllData='musicALL7.mat'
    train_xxx_ori = np.loadtxt('trainvalidtest_x_windowing_'+ str(windowsize) + '.txt')
    train_yyy_ori = np.loadtxt('trainvalidtest_y_windowing_'+ str(windowsize) + '.txt')
    #valid_xx = np.loadtxt('valid_x_windowing_'+ str(windowsize) + '.txt')
    #valid_yy = np.loadtxt('valid_y_windowing_'+ str(windowsize) + '.txt')
    #test_xx = np.loadtxt('test_x_windowing_'+ str(windowsize) + '.txt')
    #test_yy = np.loadtxt('test_y_windowing_'+ str(windowsize) + '.txt')
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
    INPUT_DIM = train_xxx_ori.shape[1]
    OUTPUT_DIM = train_yyy_ori.shape[1]
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer)+'layer'+str(nodes)+'DNN' + 'window_size' + str(windowsize) + 'training_data'+ str(portion) + '_bass_voice'
    print('Loading data...')
    print('original train_xx shape:', train_xxx_ori.shape)
    print('original train_yy shape:', train_yyy_ori.shape)
    #print('valid_xx shape:', valid_xx.shape)
    #print('valid_yy shape:', valid_yy.shape)
    #print('test_xx shape:', test_xx.shape)
    #print('test_yy shape:', test_yy.shape)
    print('Build model...')
    cvscores = []
    cvscores_test = []
    cv_log = open('cv_log+' + MODEL_NAME + '.txt', 'w')
    for i in range(9):  # add test set to share another 10%, only validate 9 times!
        train_xx, train_yy, valid_xx, valid_yy, test_xx, test_yy = divide_training_data(10, portion, i, train_xxx_ori, train_yyy_ori)
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
            model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh'))
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
        # plot(model, to_file='model.png') # draw fig of accuracy
        # score, acc = model.evaluate(valid_xx, valid_yy,
        # batch_size=batch_size,
        # show_accuracy=True)
        # print('Test score:', score)
        # print('Test accuracy:', acc)
        model = load_model(MODEL_NAME + ".hdf5")
        scores = model.evaluate(valid_xx, valid_yy, verbose=0)
        scores_test = model.evaluate(test_xx, test_yy, verbose=0)
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
    # visualize the result and put into file
    '''predict_y = model.predict(test_xx, verbose=0)
    list_of_chords = get_chord_list(predict_y.shape[1], 0)
    list_of_chords.append('other')
    fileName, numSalamiSlices = get_predict_file_name()
    sum = 0
    for i in range(len(numSalamiSlices)):
        sum += numSalamiSlices[i]
    #input(sum)
    #input(predict_y.shape[0])


    length = len(fileName)
    a_counter = 0
    for i in range(length):
        f = open('predicted_result_' + fileName[i] + '.txt', 'w')
        num_salami_slice = numSalamiSlices[i]
        for j in range(num_salami_slice):
            pointer = np.argmax(predict_y[a_counter])
            currentchord = list_of_chords[pointer]
            if(j%10==0):
                print(currentchord, end='\n', file=f)
            else:
                print(currentchord, end=' ', file=f)
            a_counter += 1
        f.close()

    #np.savetxt('predict_y_windowing_1' + '.txt', predict_yy, fmt='%.1e')'''
def evaluate_multi_label(model, x,y):
    """
    Calculate score for multi label (I do not trust Keras' version)
    :param x:
    :param y:
    :return:
    """
    yyy = model.predict(x, verbose=0)
    yy = model.predict(x, verbose=0)
    for i in yy:
        for j, item in enumerate(i):
            if(item>0.5):
                i[j] = 1
            else:
                i[j] = 0
    correctnum = 0
    fakecorrectnum = 0  # check whether this is the same with Keras' built in function
    fakecorrectnumall = 0
    for i, item in enumerate(yy):
        sign = 1
        gt = y[i]
        predition = yy[i]
        xsonority = x[i][14:26]  # need to change it adding more features or adding windows!
        for j, item2 in enumerate(item):
            if item2 != gt[j]:
                sign = 0  # wrong
                break
        if(sign == 1):
            correctnum += 1
        for j in range(12):
            if(xsonority[j] == 1):
                fakecorrectnumall += 1
                if(y[i][j] == yy[i][j]):
                    fakecorrectnum += 1
    fakecorrectrate = fakecorrectnum/fakecorrectnumall
    correctrate = correctnum / len(y)
    return correctrate, fakecorrectrate


def evaluate_f1score(model, x,y, modelname):
    mini = 0.000000001
    yyy = model.predict(x, verbose=0)
    yy = model.predict(x, verbose=0)
    if modelname != 'DNN' or modelname != 'SVM':  # need to reshape the output
       yy = yyy.reshape(yyy.shape[0]*yyy.shape[1], yyy.shape[2])
       y = y.reshape(y.shape[0]*y.shape[1], y.shape[2])
    for i in yy:
        #print(i)
        for j, item in enumerate(i):
            if (item > 0.5):
                i[j] = 1
            else:
                i[j] = 0


    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, item in enumerate(yy):
        gt = y[i]
        for j, item2 in enumerate(item):
            if item2 == 1 and gt[j] == 1:
                tp += 1
            elif item2 == 1 and gt[j] == 0:
                fp += 1
            elif item2 == 0 and gt[j] == 1:
                fn += 1
            elif item2 == 0 and gt[j] == 0:
                tn += 1
            else:
                input('no tp fp fn tn?')
    precision = tp/(tp+fp+mini)
    recall = tp/(tp+fn+mini)
    f1 = 2*precision*recall/(precision+recall+mini)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    return precision, recall, f1, accuracy, tp, fp, fn, tn


def FineTuneDNN_non_chord_tone(layer, nodes, windowsize, portion, modelID, ts):
    batch_size = 256
    epochs = 200
    patience = 20
    for i in range(0, 1):
        bootstrap = i
        extension2 = 'batch_size' + str(batch_size) + 'epochs' + str(epochs) + 'patience' + str(patience) + 'bootstrap' + str(bootstrap)
        print('Loading data...')
        extension = 'y4_non-chord_tone_pitch_class_New_annotation_12keys_music21_147'
        train_xxx_ori = np.loadtxt('.\\data_for_ML\\' + 'melodic_x_windowing_'+ str(windowsize) + extension + '.txt')
        train_yyy_ori = np.loadtxt('.\\data_for_ML\\' + 'melodic_y_windowing_'+ str(windowsize) + extension + '.txt')
        train_xxx_ori_bootstrap = train_xxx_ori
        train_yyy_ori_bootstrap = train_yyy_ori

        for i in range(bootstrap):
            if(i == 0):
                train_xxx_ori_bootstrap = np.concatenate((train_xxx_ori_bootstrap, train_xxx_ori))
                train_yyy_ori_bootstrap = np.concatenate((train_yyy_ori_bootstrap, train_yyy_ori))
            else:
                train_xxx_ori_bootstrap = np.vstack((train_xxx_ori_bootstrap, train_xxx_ori))
                train_yyy_ori_bootstrap = np.vstack((train_yyy_ori_bootstrap, train_yyy_ori))
        print('bootstrap x shape:', train_xxx_ori_bootstrap.shape)
        print('bootstrap y shape:', train_yyy_ori_bootstrap.shape)
        train_xxx_ori = train_xxx_ori_bootstrap
        train_yyy_ori = train_yyy_ori_bootstrap
        timestep = ts
        INPUT_DIM = train_xxx_ori.shape[1]
        OUTPUT_DIM = train_yyy_ori.shape[1]
        HIDDEN_NODE = nodes

        MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + str(
                windowsize) + 'training_data' + str(portion) + 'timestep' + str(timestep) + extension + extension2
        print('Loading data...')
        print('original train_xx shape:', train_xxx_ori.shape)
        print('original train_yy shape:', train_yyy_ori.shape)
        #print('valid_xx shape:', valid_xx.shape)
        #print('valid_yy shape:', valid_yy.shape)
        #print('test_xx shape:', test_xx.shape)
        #print('test_yy shape:', test_yy.shape)
        print('Build model...')
        pre = []
        pre_test = []
        rec = []
        rec_test = []
        f1 = []
        f1_test = []
        acc = []
        acc_test = []
        cvscores = []
        cvscores_test = []
        tp = []
        tn = []
        fp = []
        fn = []
        cv_log = open('.\\ML_result\\' +'cv_log+' + MODEL_NAME + '.txt', 'w')
        for i in range(3):  # add test set to share another 10%, only validate 9 times!
            train_xx, train_yy, valid_xx, valid_yy, test_xx, test_yy = divide_training_data(10, portion, i, train_xxx_ori, train_yyy_ori)
            print('Shape for cross validation...')
            print('train_xx shape:', train_xx.shape)
            print('train_yy shape:', train_yy.shape)
            print('valid_xx shape:', valid_xx.shape)
            print('valid_yy shape:', valid_yy.shape)
            print('test_xx shape:', test_xx.shape)
            print('test_yy shape:', test_yy.shape)
            if modelID != 'DNN':
                batch = timestep
                train_xx, train_yy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, timestep, train_xx, train_yy)
                valid_xx, valid_yy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, timestep, valid_xx, valid_yy)
                test_xx, test_yy = format_sequence_data(INPUT_DIM, OUTPUT_DIM, timestep, test_xx, test_yy)
                train_xx = TwoToThree.TwoToThree(train_xx, int(train_xx.shape[0] / batch), int(batch), INPUT_DIM)
                train_yy = TwoToThree.TwoToThree(train_yy, int(train_yy.shape[0] / batch), int(batch), OUTPUT_DIM)
                valid_xx = TwoToThree.TwoToThree(valid_xx, int(valid_xx.shape[0] / batch), int(batch), INPUT_DIM)
                valid_yy = TwoToThree.TwoToThree(valid_yy, int(valid_yy.shape[0] / batch), int(batch), OUTPUT_DIM)
                test_xx = TwoToThree.TwoToThree(test_xx, int(test_xx.shape[0] / batch), int(batch), INPUT_DIM)
                test_yy = TwoToThree.TwoToThree(test_yy, int(test_yy.shape[0] / batch), int(batch), OUTPUT_DIM)
            model = Sequential()
            # model.add(Embedding(36, 256, input_length=batch))
            if modelID == 'DNN':
                model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh', input_dim=INPUT_DIM))
                model.add(Dropout(0.2))
                for i in range(layer - 1):
                    # model.add(LSTM(output_dim=48, init='glorot_uniform', inner_init='orthogonal', activation='softmax', inner_activation='tanh'))  # try using a GRU instead, for fun
                    # model.add(LSTM(input_dim=INPUT_DIM, output_dim=500, return_sequences=True, init='glorot_uniform'))
                    # model.add(LSTM(output_dim=500, return_sequences=True))
                    # model.add(LSTM(output_dim=500, return_sequences=True))
                    # model.add(LSTM(48))
                    model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh'))
                    model.add(Dropout(0.2))
            elif modelID == 'BLSTM':
                print("fuck you shape: ", train_xx.shape, train_yy.shape)
                model.add(Bidirectional(
                    LSTM(return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_dim=INPUT_DIM, units=HIDDEN_NODE,
                         kernel_initializer="glorot_uniform"), input_shape=train_xx.shape))
                for i in range(layer - 1):
                    model.add(
                        Bidirectional(LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
            elif modelID == 'RNN':
                print("fuck you shape: ", train_xx.shape, train_yy.shape)
                model.add(SimpleRNN(input_dim=INPUT_DIM, units=HIDDEN_NODE, return_sequences=True, dropout=0.2,recurrent_dropout=0.2))
                for i in range(layer - 1):
                    model.add(
                        SimpleRNN(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            elif modelID == 'LSTM':
                print("fuck you shape: ", train_xx.shape, train_yy.shape)
                model.add(
                    LSTM(return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_dim=INPUT_DIM, units=HIDDEN_NODE,
                         kernel_initializer="glorot_uniform"))#, input_shape=train_xx.shape)
                for i in range(layer - 1):
                    model.add(
                        LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            if modelID == 'DNN':
                model.add(Dense(OUTPUT_DIM, init='uniform'))
            else:
                model.add(TimeDistributed(Dense(OUTPUT_DIM)))
            # model.add(Dropout(0.5)) # dropout does not add at output layer!!
            model.add(Activation('sigmoid'))  # need time distributed softmax??

            # try using different optimizers and different optimizer configs
            #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            #sgd = SGD(lr=0.1, decay=0.002, momentum=0.5, nesterov=False)  # lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
            model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['binary_accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)  # set up early stopping
            print("Train...")
            checkpointer = ModelCheckpoint(filepath='.\\data_for_ML\\' + MODEL_NAME + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
            # hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=3,validation_split=0.2, shuffle=True, verbose=1, show_accuracy=True, callbacks=[early_stopping])
            hist = model.fit(train_xx, train_yy, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                             validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer])  # for debug
            # plot(model, to_file='model.png') # draw fig of accuracy
            # score, acc = model.evaluate(valid_xx, valid_yy,
            # batch_size=batch_size,
            # show_accuracy=True)
            # print('Test score:', score)
            # print('Test accuracy:', acc)
            model = load_model('.\\data_for_ML\\' + MODEL_NAME + ".hdf5")
            #plot_model(model, to_file=MODEL_NAME + 'png')
            scores = model.evaluate(valid_xx, valid_yy, verbose=0)
            scores_test = model.evaluate(test_xx, test_yy, verbose=0)
            print(' valid_acc: ', scores[1])
            cvscores.append(scores[1] * 100)
            cvscores_test.append(scores_test[1] * 100)
            # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)

            precision, recall, f1score, accuracy, true_positive, false_positive, false_negative, true_negative = evaluate_f1score(model, valid_xx, valid_yy, modelID)
            precision_test, recall_test, f1score_test, accuracy_test, asd, sdf, dfg, fgh= evaluate_f1score(model, test_xx, test_yy, modelID)
            pre.append(precision*100)
            pre_test.append(precision_test * 100)
            rec.append(recall * 100)
            rec_test.append(recall_test * 100)
            f1.append(f1score * 100)
            f1_test.append(f1score_test * 100)
            acc.append(accuracy * 100)
            acc_test.append(accuracy_test * 100)
            tp.append(true_positive)
            fp.append(false_positive)
            fn.append(false_negative)
            tn.append(true_negative)
        print(np.mean(cvscores), np.std(cvscores))
        print(MODEL_NAME, file=cv_log)
        model = load_model(MODEL_NAME + ".hdf5")
        #print(model.summary(), file=cv_log)

        model.summary(print_fn=lambda x: cv_log.write(x + '\n'))  #output model struc ture into the text file
        print('valid accuracy:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
        print('valid precision:', np.mean(pre), '%', '±', np.std(pre), '%', file=cv_log)
        print('valid recall:', np.mean(rec), '%', '±', np.std(rec), '%', file=cv_log)
        print('valid f1:', np.mean(f1), '%', '±', np.std(f1), '%', file=cv_log)
        print('valid acc (validate previous):', np.mean(acc), '%', '±', np.std(acc), '%', file=cv_log)
        print('valid tp number:', np.mean(tp), '±', np.std(tp), file=cv_log)
        print('valid fp number:', np.mean(fp), '±', np.std(fp), file=cv_log)
        print('valid fn number:', np.mean(fn), '±', np.std(fn), file=cv_log)
        print('valid tn number:', np.mean(tn), '±', np.std(tn), file=cv_log)
        for i in range(len(cvscores_test)):
            #print('Test accuracy:', i, cvscores_test[i], '%', file=cv_log)
            #print('Test precision:', i, pre_test[i], '%', file=cv_log)
            #print('Test reall:', i, rec_test[i], '%', file=cv_log)
            print('Test f1:', i, f1_test[i], '%', file=cv_log)
            #print('Test acc:', i, acc_test[i], '%', file=cv_log)
        print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
        print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
        print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
        print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
        print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)
    # visualize the result and put into file


def FineTuneDNN_non_chord_tone_shuffle(layer,nodes,windowsize,portion,shuffletimes):
    total_f1 = []
    total_f1_std = []
    total_acc = []
    total_acc_std = []
    for i in range(shuffletimes):
        print('Loading data...')
        extension = 'y4_non-chord_tone'
        train_xxx_ori = np.loadtxt('trainvalidtest_x_windowing_' + str(windowsize) + extension + '.txt')
        train_yyy_ori = np.loadtxt('trainvalidtest_y_windowing_' + str(windowsize) + extension + '.txt')
        print('shuffle data ' + str(i) + 'times')
        rng_state = np.random.get_state()  # shuffle two arrays with the same random seed
        np.random.shuffle(train_xxx_ori)
        np.random.set_state(rng_state)
        np.random.shuffle(train_yyy_ori)
        batch_size = 50
        INPUT_DIM = train_xxx_ori.shape[1]
        OUTPUT_DIM = train_yyy_ori.shape[1]
        HIDDEN_NODE = nodes
        MODEL_NAME = str(layer)+'layer'+str(nodes)+'DNN' + 'window_size' + str(windowsize) + 'training_data'+ str(portion) + extension + '_shuffletimes_' + str(i)
        print('Loading data...')
        print('original train_xx shape:', train_xxx_ori.shape)
        print('original train_yy shape:', train_yyy_ori.shape)
        #print('valid_xx shape:', valid_xx.shape)
        #print('valid_yy shape:', valid_yy.shape)
        #print('test_xx shape:', test_xx.shape)
        #print('test_yy shape:', test_yy.shape)
        print('Build model...')
        pre = []
        pre_test = []
        rec = []
        rec_test = []
        f1 = []
        f1_test = []
        acc = []
        acc_test = []
        cvscores = []
        cvscores_test = []
        tp = []
        tn = []
        fp = []
        fn = []
        cv_log = open('cv_log+' + MODEL_NAME + '.txt', 'w')
        for i in range(10):  # add test set to share another 10%, only validate 9 times!
            train_xx, train_yy, valid_xx, valid_yy, test_xx, test_yy = divide_training_data(10, portion, i, train_xxx_ori, train_yyy_ori)
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
                model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh'))
                model.add(Dropout(0.2))
            model.add(Dense(OUTPUT_DIM, init='uniform'))
            # model.add(Dropout(0.5)) # dropout does not add at output layer!!
            model.add(Activation('sigmoid'))  # need time distributed softmax??

            # try using different optimizers and different optimizer configs
            # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            sgd = SGD(lr=0.1, decay=0.002, momentum=0.5,
                      nesterov=False)  # lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # set up early stopping
            print("Train...")
            checkpointer = ModelCheckpoint(filepath=MODEL_NAME + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
            # hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=3,validation_split=0.2, shuffle=True, verbose=1, show_accuracy=True, callbacks=[early_stopping])
            hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=100, shuffle=True, verbose=1,
                             validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer])  # for debug
            # plot(model, to_file='model.png') # draw fig of accuracy
            # score, acc = model.evaluate(valid_xx, valid_yy,
            # batch_size=batch_size,
            # show_accuracy=True)
            # print('Test score:', score)
            # print('Test accuracy:', acc)
            model = load_model(MODEL_NAME + ".hdf5")
            '''scores, fakescores = evaluate_multi_label(model, valid_xx, valid_yy)
            scores_test, fakescores_test = evaluate_multi_label(model, test_xx, test_yy)

            print(' valid_acc: ', scores)
            print(' test_acc: ', scores_test)
            print(' note_valid_acc: ', fakescores)
            print(' note_test_acc: ', fakescores_test)
            cvscores.append(scores * 100)
            cvscores_test.append(scores_test * 100)
            fake_cvscores.append(fakescores * 100)
            fake_cvscores_test.append(fakescores_test * 100)
            # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)
        print(np.mean(cvscores), np.std(cvscores))
        print(MODEL_NAME, file=cv_log)
        print('valid:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
        print('note_valid:', np.mean(fake_cvscores), '%', '±', np.std(fake_cvscores), '%', file=cv_log)
        for i in range(len(cvscores_test)):
            print('Test:', i, cvscores_test[i], '%', file=cv_log)
            print('note_Test:', i, fake_cvscores_test[i], '%', file=cv_log)
        print('Test:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
        print('note_Test:', np.mean(fake_cvscores_test), '%', '±', np.std(fake_cvscores_test), '%', file=cv_log)'''
            scores = model.evaluate(valid_xx, valid_yy, verbose=0)
            scores_test = model.evaluate(test_xx, test_yy, verbose=0)
            print(' valid_acc: ', scores[1])
            cvscores.append(scores[1] * 100)
            cvscores_test.append(scores_test[1] * 100)
            # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)

            precision, recall, f1score, accuracy, true_positive, false_positive, false_negative, true_negative = evaluate_f1score(model, valid_xx, valid_yy)
            precision_test, recall_test, f1score_test, accuracy_test, asd, sdf, dfg, fgh= evaluate_f1score(model, test_xx, test_yy)
            pre.append(precision*100)
            pre_test.append(precision_test * 100)
            rec.append(recall * 100)
            rec_test.append(recall_test * 100)
            f1.append(f1score * 100)
            f1_test.append(f1score_test * 100)
            acc.append(accuracy * 100)
            acc_test.append(accuracy_test * 100)
            tp.append(true_positive)
            fp.append(false_positive)
            fn.append(false_negative)
            tn.append(true_negative)
        print(np.mean(cvscores), np.std(cvscores))
        print(MODEL_NAME, file=cv_log)
        print('valid accuracy:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
        print('valid precision:', np.mean(pre), '%', '±', np.std(pre), '%', file=cv_log)
        print('valid recall:', np.mean(rec), '%', '±', np.std(rec), '%', file=cv_log)
        print('valid f1:', np.mean(f1), '%', '±', np.std(f1), '%', file=cv_log)
        print('valid acc (validate previous):', np.mean(acc), '%', '±', np.std(acc), '%', file=cv_log)
        print('valid tp number:', np.mean(tp), '±', np.std(tp), file=cv_log)
        print('valid fp number:', np.mean(fp), '±', np.std(fp), file=cv_log)
        print('valid fn number:', np.mean(fn), '±', np.std(fn), file=cv_log)
        print('valid tn number:', np.mean(tn), '±', np.std(tn), file=cv_log)
        for i in range(len(cvscores_test)):
            print('Test accuracy:', i, cvscores_test[i], '%', file=cv_log)
            #print('Test precision:', i, pre_test[i], '%', file=cv_log)
            #print('Test reall:', i, rec_test[i], '%', file=cv_log)
            #print('Test f1:', i, f1_test[i], '%', file=cv_log)
            #print('Test acc:', i, acc_test[i], '%', file=cv_log)
        print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
        print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
        print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
        print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
        print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)
        total_f1.append(np.mean(f1_test))
        total_f1_std.append(np.std(f1_test))
        total_acc.append(np.mean(acc_test))
        total_acc_std.append(np.std(acc_test))
    cv_log2 = open('cv_log+' + MODEL_NAME + 'Total.txt', 'w')
    for i in range(len(total_acc)):
        print('shuffle time: ' , i + 1 , ' Test f1: ' , total_f1[i], '±', total_f1_std[i], 'Test acc: ', total_acc[i], '±', total_acc_std[i] , file=cv_log2)
    print('Total: f1: ', np.mean(total_f1), '±', np.std(total_f1), 'acc: ', np.mean(total_acc), '±', np.std(total_acc), file=cv_log2)

