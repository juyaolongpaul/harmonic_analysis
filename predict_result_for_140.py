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
from keras.layers import LSTM, Bidirectional, RNN, SimpleRNN, TimeDistributed
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
from DNN_no_window import evaluate_f1score
from get_input_and_output import determine_middle_name
def get_predict_file_name(input):
    filename = []
    num_salami_slices = []
    for id, fn in enumerate(os.listdir(input)):
        if fn.find('KB') != -1:
            if fn.find('340') != -1 or fn.find('358') != -1 or fn.find('362') != -1 or fn.find('003') != -1 or fn.find('008') != -1 or fn.find('014') != -1:

                #if fn.find('cKE') != -1:
                    filename.append(fn)

    for id, fn in enumerate(filename):
        length = 0
        s = converter.parse(input + fn)
        sChords = s.chordify()
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

            length += 1
        num_salami_slices.append(length)
    return filename, num_salami_slices


def train_and_predict_non_chord_tone(layer, nodes, windowsize, portion, modelID, ts, bootstraptime, sign, augmentation, cv, pitchclass):
    keys, music21 = determine_middle_name(augmentation, sign)
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
    batch_size = 256
    epochs = 200
    patience = 20
    extension2 = 'batch_size' + str(batch_size) + 'epochs' + str(epochs) + 'patience' + str(patience) + 'bootstrap' + str(bootstraptime)
    print('Loading data...')
    extension = 'y4_non-chord_tone_'+ pitchclass + '_New_annotation_' + keys + '_' +music21+ '_' + '147'
    train_xxx_ori = np.loadtxt('.\\data_for_ML\\' + sign +'_x_windowing_' + str(windowsize) + extension + '.txt')
    train_yyy_ori = np.loadtxt('.\\data_for_ML\\' + sign +'_y_windowing_' + str(windowsize) + extension + '.txt')
    train_xxx_ori_bootstrap = train_xxx_ori
    train_yyy_ori_bootstrap = train_yyy_ori
    for i in range(bootstraptime):
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
    MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                 str(windowsize) + 'training_data' + str(portion) + 'timestep' \
                 + str(timestep) + extension + extension2
    print('Loading data...')
    print('original train_xx shape:', train_xxx_ori.shape)
    print('original train_yy shape:', train_yyy_ori.shape)
    print('Build model...')
    for times in range(0, cv + 1):
        train_xx, train_yy, valid_xx, valid_yy, rubbish_x, rubbish_y = divide_training_data(10, portion, times, train_xxx_ori, train_yyy_ori, testset='N')
        cv_log = open('.\\ML_result\\' + 'cv_log+' + MODEL_NAME + 'predict.txt', 'w')  # only have valid result
        if not (os.path.isfile(('.\\ML_result\\' + MODEL_NAME + ".hdf5"))):
            print('training and predicting...')
            print('train_xx shape:', train_xx.shape)
            print('train_yy shape:', train_yy.shape)
            print('valid_xx shape:', valid_xx.shape)
            print('valid_yy shape:', valid_yy.shape)
            model = Sequential()
            # model.add(Embedding(36, 256, input_length=batch))
            if modelID == 'DNN':
                model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh', input_dim=INPUT_DIM))
                model.add(Dropout(0.2))
                for i in range(layer - 1):
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
                model.add(SimpleRNN(input_dim=INPUT_DIM, units=HIDDEN_NODE, return_sequences=True, dropout=0.2,
                                    recurrent_dropout=0.2))
                for i in range(layer - 1):
                    model.add(
                        SimpleRNN(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            elif modelID == 'LSTM':
                print("fuck you shape: ", train_xx.shape, train_yy.shape)
                model.add(
                    LSTM(return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_dim=INPUT_DIM, units=HIDDEN_NODE,
                         kernel_initializer="glorot_uniform"))  # , input_shape=train_xx.shape)
                for i in range(layer - 1):
                    model.add(
                        LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
            if modelID == 'DNN':
                model.add(Dense(OUTPUT_DIM, init='uniform'))
            else:
                model.add(TimeDistributed(Dense(OUTPUT_DIM)))
            model.add(Activation('sigmoid'))
            model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)  # set up early stopping
            print("Train...")
            checkpointer = ModelCheckpoint(filepath='.\\ML_result\\' + MODEL_NAME + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
            hist = model.fit(train_xx, train_yy, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                             validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer])
        # visualize the result and put into file
        test_xx = np.loadtxt('.\\data_for_ML\\' +sign +'_x_windowing_' + str(windowsize) + extension[:-3] + '6.txt')
        test_yy = np.loadtxt('.\\data_for_ML\\' +sign +'_y_windowing_' + str(windowsize) + extension[:-3] + '6.txt')
        model = load_model('.\\ML_result\\' + MODEL_NAME + ".hdf5")
        predict_y = model.predict(test_xx, verbose=0)
        scores = model.evaluate(valid_xx, valid_yy, verbose=0)
        scores_test = model.evaluate(test_xx, test_yy, verbose=0)
        print(' valid_acc: ', scores[1])
        cvscores.append(scores[1] * 100)
        cvscores_test.append(scores_test[1] * 100)
        # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)

        precision, recall, f1score, accuracy, true_positive, false_positive, false_negative, true_negative = evaluate_f1score(
            model, valid_xx, valid_yy, modelID)
        precision_test, recall_test, f1score_test, accuracy_test, asd, sdf, dfg, fgh = evaluate_f1score(model, test_xx,
                                                                                                        test_yy, modelID)
        pre.append(precision * 100)
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
    model = load_model('.\\ML_result\\' + MODEL_NAME + ".hdf5")
    # print(model.summary(), file=cv_log)

    model.summary(print_fn=lambda x: cv_log.write(x + '\n'))  # output model struc ture into the text file
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
        # print('Test accuracy:', i, cvscores_test[i], '%', file=cv_log)
        # print('Test precision:', i, pre_test[i], '%', file=cv_log)
        # print('Test reall:', i, rec_test[i], '%', file=cv_log)
        print('Test f1:', i, f1_test[i], '%', file=cv_log)
        # print('Test acc:', i, acc_test[i], '%', file=cv_log)
    print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
    print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
    print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
    print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
    print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)
    for i in predict_y: # regulate the prediction
        for j, item in enumerate(i):
            if (item > 0.5):
                i[j] = 1
            else:
                i[j] = 0
    input = '.\\bach-371-chorales-master-kern\\kern\\'
    fileName, numSalamiSlices = get_predict_file_name(input)
    sum = 0
    for i in range(len(numSalamiSlices)):
        sum += numSalamiSlices[i]
    # input(sum)
    # input(predict_y.shape[0])


    length = len(fileName)
    a_counter = 0
    a_counter_correct = 0
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    for i in range(length):
        f = open('.\\predicted_result\\' + 'predicted_result_' + fileName[i] + '_non-chord_tone' + '.txt', 'w')
        num_salami_slice = numSalamiSlices[i]
        correct_num = 0
        for j in range(num_salami_slice):
            gt = test_yy[a_counter]
            prediction = predict_y[a_counter]
            correct_bit = 0
            for i in range(len(gt)):
                if (gt[i] == prediction[i]):  # the label is correct
                    correct_bit += 1
            if(correct_bit == len(gt)):
                correct_num += 1
            else:
                print('error')
            nonchordpitchclassptr = [-1] * 4
            yyptr = -1
            dimension = test_xx.shape[1]
            realdimension = int(dimension/(2*windowsize+1))
            x = test_xx[a_counter][realdimension*windowsize:realdimension*(windowsize+1)]
            for i in range(len(x) - 2):
                if (x[i] == 1):  # non-chord tone
                    yyptr += 1
                    if (prediction[yyptr] == 1):

                        nonchordpitchclassptr[yyptr] = i % 12

            if (nonchordpitchclassptr == [-1] * 4):
                print('n/a', end=' ', file=f)
            else:
                for item in nonchordpitchclassptr:
                    if (item != -1):
                        print(pitchclass[item], end='', file=f)
                print(end=' ', file=f)
            a_counter += 1
        a_counter_correct += correct_num
        print(end='\n', file=f)
        print('accucary: ' + str(correct_num/num_salami_slice), end='\n', file=f)
        print('num of correct answers: ' + str(correct_num) + ' number of salami slices: ' + str(num_salami_slice), file=f)
        print('accumulative accucary: ' + str(a_counter_correct / a_counter), end='\n', file=f)
        f.close()
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
    checkpointer = ModelCheckpoint(filepath='.\\data_for_ML\\' + MODEL_NAME + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
    # hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=3,validation_split=0.2, shuffle=True, verbose=1, show_accuracy=True, callbacks=[early_stopping])
    hist = model.fit(train_xx, train_yy, batch_size=batch_size, nb_epoch=100, shuffle=True, verbose=1,
                     validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer])  # for debug

    # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)

    # visualize the result and put into file
    model = load_model('.\\data_for_ML\\' + MODEL_NAME + ".hdf5")
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
    train_and_predict_non_chord_tone(2, 200, 2, 1, 'DNN', 10)