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
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
import h5py
import re
import os
import numpy as np
import keras.callbacks as CB
import sys
import string
import time
import SaveModelLog
from get_input_and_output import get_chord_list, get_chord_line, calculate_freq
from music21 import *
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from DNN_no_window_cross_validation import divide_training_data
from DNN_no_window import evaluate_f1score
from get_input_and_output import determine_middle_name, find_id, get_id, determine_middle_name2
def get_predict_file_name(input, data_id, augmentation):
    filename = []
    num_salami_slices = []
    for id, fn in enumerate(os.listdir(input)):
        if fn.find('KB') != -1:
            p = re.compile(r'\d{3}')  # find 3 digit in the file name
            id_id = p.findall(fn)
            if id_id[0] in data_id:  # if the digit found in the list, add this file

                if (augmentation != 'Y'):  # Don't want data augmentation in 12 keys
                    if (fn.find('cKE') != -1 or fn.find('c_oriKE') != -1):  # only wants key c
                        filename.append(fn)
                else:
                    filename.append(fn)



    for id, fn in enumerate(filename):
        length = 0
        s = converter.parse(os.path.join(input, fn))
        sChords = s.chordify()
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

            length += 1
        num_salami_slices.append(length)
    return filename, num_salami_slices


def binary_decode(arr):
    """
    Translate binary encoding into decimal
    :param arr:
    :return:
    """
    arr_decoded = []
    for i, item in enumerate(arr):
        total = 0
        for index, val in enumerate(reversed(item)):
            total += (val * 2**index)
        arr_decoded.append(int(total))

    return arr_decoded


def binary_encode(arr):
    """
    Translate decimal into binary
    :param arr:
    :return:
    """
    arr_encoded = []
    for i, item in enumerate(arr):
        row = np.array(list(np.binary_repr(item).zfill(4))).astype(float)
        # https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        if i == 0:
            arr_encoded = np.concatenate((arr_encoded, row))
        else:
            arr_encoded = np.vstack((arr_encoded, row))
    return arr_encoded


def onehot_decode(arr):
    """
    Translate onehot encoding into decimal
    :param arr:
    :return:
    """
    arr_decoded = []
    for i, item in enumerate(arr):
        for ii, itemitem in enumerate(item):
            if itemitem == 1:
                arr_decoded.append(ii)

    return arr_decoded


def onehot_encode(arr, dim):
    """
    Translate int into one hot encoding
    :param arr:
    :return:
    """
    arr_encoded = []
    for i, item in enumerate(arr):
        print('progress:',i,'/',arr.shape[0])
        row = [0] * dim
        # https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        row[item] = 1
        arr_encoded.append(row)
    return arr_encoded



def bootstrap_data(x, y, times):
    """
    bootstraping data
    :param x:
    :param y:
    :param times:
    :return:
    """
    xx = x
    yy = y
    for i in range(times):
        xx = np.vstack((xx, x))
        yy = np.vstack((yy, y))
    return xx, yy


def train_and_predict_non_chord_tone(layer, nodes, windowsize, portion, modelID, ts, bootstraptime, sign, augmentation, cv, pitch_class, ratio, input, output, distributed, balanced, outputtype):
    id_sum = find_id(output, distributed)  # get 3 digit id of the chorale
    num_of_chorale = len(id_sum)
    train_num = num_of_chorale - int((num_of_chorale * (1 - ratio)/2))*2
    #train_num = int(num_of_chorale * ratio)
    test_num = int((num_of_chorale - train_num) / 2)
    #keys, music21 = determine_middle_name(augmentation, sign, portion)
    keys, keys1, music21 = determine_middle_name2(augmentation, sign, pitch_class)
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
    epochs = 500
    patience = 50
    extension2 = 'batch_size' + str(batch_size) + 'epochs' + str(epochs) + 'patience' + str(patience) + 'bootstrap' + str(bootstraptime) + 'balanced' + str(balanced)
    print('Loading data...')
    extension = sign + outputtype+ pitch_class + '_New_annotation_' + keys + '_' +music21+ '_' + 'training' + str(train_num)
    #train_xxx_ori = np.loadtxt('.\\data_for_ML\\' + sign +'_x_windowing_' + str(windowsize) + extension + '.txt')
    #train_yyy_ori = np.loadtxt('.\\data_for_ML\\' + sign +'_y_windowing_' + str(windowsize) + extension + '.txt')
    timestep = ts
    #INPUT_DIM = train_xxx_ori.shape[1]
    #OUTPUT_DIM = train_yyy_ori.shape[1]
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                 str(windowsize) + 'training_data' + str(portion) + 'timestep' \
                 + str(timestep) + extension + extension2
    print('Loading data...')
    #print('original train_xx shape:', train_xxx_ori.shape)
    #print('original train_yy shape:', train_yyy_ori.shape)
    print('Build model...')
    cv_log = open(os.path.join('.', 'ML_result', 'cv_log+') + MODEL_NAME + 'predict.txt', 'w')
    csv_logger = CSVLogger(os.path.join('.', 'ML_result', 'cv_log+') + MODEL_NAME + 'predict_log.csv', append=True, separator=';')
    for times in range(cv):
        MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                     str(windowsize) + 'training_data' + str(portion) + 'timestep' \
                     + str(timestep) + extension + extension2 + '_cv_' + str(times + 1)
        train_id, valid_id, test_id = get_id(id_sum, num_of_chorale, times)
        train_num = len(train_id)
        valid_num = len(valid_id)
        test_num = len(test_id)

        #train_xx, train_yy, valid_xx, valid_yy, rubbish_x, rubbish_y = divide_training_data(10, portion, times, train_xxx_ori, train_yyy_ori, testset='N')
         # only have valid result
        valid_xx = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_x_windowing_' + str(
            windowsize) + outputtype + pitch_class + '_New_annotation_' + keys1 + '_' + music21 + '_' + 'validing' + str(
            valid_num) + '_cv_' + str(times + 1) + '.txt')
        valid_yy = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_y_windowing_' + str(
            windowsize) + outputtype + pitch_class + '_New_annotation_' + keys1 + '_' + music21 + '_' + 'validing' + str(
            valid_num) + '_cv_' + str(times + 1) + '.txt')
        if not (os.path.isfile((os.path.join('.', 'ML_result', MODEL_NAME) + ".hdf5"))):
            train_xx = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_x_windowing_' + str(
                windowsize) + outputtype + pitch_class + '_New_annotation_' + keys + '_' + music21 + '_' + 'training' + str(
                train_num) + '_cv_' + str(times + 1) + '.txt')
            train_yy = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_y_windowing_' + str(
                windowsize) + outputtype + pitch_class + '_New_annotation_' + keys + '_' + music21 + '_' + 'training' + str(
                train_num) + '_cv_' + str(times + 1) + '.txt')
            INPUT_DIM = train_xx.shape[1]
            OUTPUT_DIM = train_yy.shape[1]

            train_xx, train_yy = bootstrap_data(train_xx, train_yy, bootstraptime)
            train_xx = train_xx[
                      :int(portion * train_xx.shape[0])]  # expose the option of training only on a subset of data
            train_yy = train_yy[:int(portion * train_yy.shape[0])]
            if balanced: # re-balance the data
                if outputtype == "NCT":
                    # http://imbalanced-learn.org/en/stable/introduction.html#problem-statement-regarding-imbalanced-data-sets
                    train_yy_encoded = binary_decode(train_yy)
                    ros = RandomOverSampler(ratio='minority')
                    train_xx_imbalanced = train_xx
                    train_yy_imbalanced = train_yy
                    ros_statistics = ros.fit(train_xx, train_yy_encoded)
                    train_xx, train_yy_balanced = ros.fit_sample(train_xx, train_yy_encoded)

                    train_xx = train_xx[:int(1.5*train_xx_imbalanced.shape[0])]
                    train_yy_balanced = train_yy_balanced[:int(1.5 * train_yy_imbalanced.shape[0])]
                    train_yy = binary_encode(train_yy_balanced)
                else:
                    ros = RandomOverSampler()
                    train_xx_imbalanced = train_xx
                    train_yy_imbalanced = train_yy
                    train_yy_encoded = onehot_decode(train_yy)
                    train_xx, train_yy_balanced = ros.fit_sample(train_xx, train_yy_encoded)
                    train_yy = onehot_encode(train_yy_balanced, train_yy_imbalanced.shape[1])
                    train_yy = np.asarray(train_yy)
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
            if outputtype == "NCT":
                model.add(Activation('sigmoid'))
                model.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            elif outputtype == "CL":
                model.add(Activation('softmax'))
                model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)  # set up early stopping
            print("Train...")
            checkpointer = ModelCheckpoint(filepath=os.path.join('.','ML_result', MODEL_NAME) + ".hdf5", verbose=1, save_best_only=True, monitor='val_loss')
            hist = model.fit(train_xx, train_yy, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                             validation_data=(valid_xx, valid_yy), callbacks=[early_stopping, checkpointer, csv_logger])
        # visualize the result and put into file
        test_xx = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_x_windowing_' + str(
            windowsize) + outputtype + pitch_class + '_New_annotation_' + keys1 + '_' + music21 + '_' + 'testing' + str(
            test_num) + '_cv_' + str(times  + 1) + '.txt')
        test_yy = np.loadtxt(os.path.join('.', 'data_for_ML', sign) + '_y_windowing_' + str(
            windowsize) + outputtype + pitch_class + '_New_annotation_' + keys1 + '_' + music21 + '_' + 'testing' + str(
            test_num) + '_cv_' + str(times  + 1) + '.txt')
        model = load_model(os.path.join('.','ML_result', MODEL_NAME) + ".hdf5")
        test_yy_int = np.asarray(onehot_decode(test_yy))
        predict_y = model.predict_classes(test_xx, verbose=0)
        scores = model.evaluate(valid_xx, valid_yy, verbose=0)
        scores_test = model.evaluate(test_xx, test_yy, verbose=0)
        print(' valid_acc: ', scores[1])
        cvscores.append(scores[1] * 100)
        cvscores_test.append(scores_test[1] * 100)
        # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)
        with open('chord_name.txt') as f:
            chord_name = f.read().splitlines()
        with open('chord_name.txt') as f:
            chord_name2 = f.read().splitlines() # delete all the chords which do not appear in the test set
        # print(matrix, file=cv_log)
        for i, item in enumerate(chord_name):
            if i not in test_yy_int and i not in predict_y:
                chord_name2.remove(item)
        matrix = confusion_matrix(test_yy_int, predict_y)
        print(classification_report(test_yy_int, predict_y, target_names=chord_name2), file=cv_log)
        if outputtype == "NCT":
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
        # prediction put into files
        fileName, numSalamiSlices = get_predict_file_name(input, test_id, 'N')
        sum = 0
        for i in range(len(numSalamiSlices)):
            sum += numSalamiSlices[i]
        # input(sum)
        # input(predict_y.shape[0])

        length = len(fileName)
        a_counter = 0
        a_counter_correct = 0
        if os.path.isfile('.\\predicted_result\\' + 'predicted_result_' + 'ALTOGETHER' + outputtype + sign + pitch_class + '.txt'):
            f_all = open(
            '.\\predicted_result\\' + 'predicted_result_' + 'ALTOGETHER' + outputtype + sign + pitch_class + '.txt',
            'a') # create this file to track every type of mistakes
        else:
            f_all = open(
                '.\\predicted_result\\' + 'predicted_result_' + 'ALTOGETHER' + outputtype + sign + pitch_class + '.txt',
                'w')  # create this file to track every type of mistakes
        for i in range(length):
            print(fileName[i][:-4], file=f_all)
            num_salami_slice = numSalamiSlices[i]
            correct_num = 0
            s = converter.parse(os.path.join(input, fileName[i]))  # the source musicXML file
            sChords = s.chordify()
            s.insert(0, sChords)
            for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                thisChord.addLyric(chord_name[test_yy_int[a_counter]])
                thisChord.addLyric(chord_name[predict_y[a_counter]])
                thisChord.closedPosition(forceOctave=4, inPlace=True)
                if test_yy_int[a_counter] == predict_y[a_counter]:
                    correct_num += 1
                    print(chord_name[predict_y[a_counter]], end=' ', file=f_all)
                else:
                    print(chord_name[test_yy_int[a_counter]] + '->' + chord_name[predict_y[a_counter]], end=' ', file=f_all)
                a_counter += 1
            a_counter_correct += correct_num
            print(end='\n', file=f_all)
            print('accucary: ' + str(correct_num / num_salami_slice), end='\n', file=f_all)
            print('num of correct answers: ' + str(correct_num) + ' number of salami slices: ' + str(num_salami_slice),
                  file=f_all)
            print('accumulative accucary: ' + str(a_counter_correct / a_counter), end='\n', file=f_all)
            s.write('musicxml',
                    fp='.\\predicted_result\\' + 'predicted_result_' + fileName[i][:-4] + outputtype + sign + pitch_class + '.xml')
            # output result in musicXML
        f_all.close()


    print(np.mean(cvscores), np.std(cvscores))
    print(MODEL_NAME, file=cv_log)
    model = load_model(os.path.join('.','ML_result', MODEL_NAME) + ".hdf5")
    model.summary(print_fn=lambda x: cv_log.write(x + '\n'))  # output model struc ture into the text file
    print('valid accuracy:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
    if outputtype == 'NCT':
        print('valid precision:', np.mean(pre), '%', '±', np.std(pre), '%', file=cv_log)
        print('valid recall:', np.mean(rec), '%', '±', np.std(rec), '%', file=cv_log)
        print('valid f1:', np.mean(f1), '%', '±', np.std(f1), '%', file=cv_log)
        print('valid acc (validate previous):', np.mean(acc), '%', '±', np.std(acc), '%', file=cv_log)
        print('valid tp number:', np.mean(tp), '±', np.std(tp), file=cv_log)
        print('valid fp number:', np.mean(fp), '±', np.std(fp), file=cv_log)
        print('valid fn number:', np.mean(fn), '±', np.std(fn), file=cv_log)
        print('valid tn number:', np.mean(tn), '±', np.std(tn), file=cv_log)
        for i in range(len(cvscores_test)):
            print('Test f1:', i, f1_test[i], '%', file=cv_log)
    elif outputtype == "CL":
        for i in range(len(cvscores_test)):
            print('Test acc:', i, cvscores_test[i], '%', file=cv_log)
    print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
    if outputtype == 'NCT':
        print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
        print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
        print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
        print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)



if __name__ == "__main__":
    train_and_predict_non_chord_tone(2, 200, 2, 1, 'DNN', 10)