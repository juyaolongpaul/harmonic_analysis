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
import collections
#import graphviz
np.random.seed(1337)  # for reproducibility
from keras import optimizers
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as TREE
from keras.preprocessing import sequence
from keras.utils import np_utils
# from keras.utils.visualize_util import plot # draw fig
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Bidirectional, RNN, SimpleRNN, TimeDistributed
from keras.datasets import imdb
from scipy.io import loadmat
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from collections import Counter
from keras.callbacks import TensorBoard
import itertools
import re
import os
import numpy as np
#import SaveModelLog
from get_input_and_output import get_chord_list, get_chord_line, calculate_freq
from get_input_and_output import determine_NCT, fill_in_pitch_class, contain_concert_pitch, contain_chordify_voice
from music21 import *
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
#from DNN_no_window_cross_validation import divide_training_data
from DNN_no_window import evaluate_f1score
from get_input_and_output import determine_middle_name, find_id, find_id_FB, get_id, determine_middle_name2
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from test_musicxml_gt import translate_chord_name_into_music21
from keras_self_attention import SeqSelfAttention
from get_input_and_output import adding_window_one_hot
from sklearn.metrics import accuracy_score
from transpose_to_C_chords import transpose
from get_input_and_output import get_pitch_class_for_four_voice, get_bass_note
from FB2lyrics import is_suspension, colllapse_interval

c2 = ['c', 'd-', 'd', 'e-', 'e', 'f', 'f#', 'g', 'a-', 'a', 'b-', 'b']


def find_tranposed_interval(fn):
    """
    Get the key info from the file name, transpose back to the original key
    :param fn:
    :return:
    """
    ptr = fn.find('KB')
    if '-' not in fn and '#' not in fn:  # keys do not have accidentals
        key_info = fn[ptr + 2]
    else:
        key_info = fn[ptr + 2: ptr + 4]
    if key_info.isupper():  # major key
        mode = 'major'
    else:
        mode = 'minor'
    if mode == 'minor':
        transposed_interval = interval.Interval(pitch.Pitch('A'), pitch.Pitch(key_info))
    else:
        transposed_interval = interval.Interval(pitch.Pitch('C'), pitch.Pitch(key_info))
    return transposed_interval, key_info


def transpose_chord(transposed_interval, chord):
    acc = re.compile(r'[#-]+')
    id_id = acc.findall(chord)
    if id_id != []:  # has flat or sharp
        root_ptr = re.search(r'[#-]+', chord).end()  # get the last flat or sharp position
        transposed_result = transpose(chord[0: root_ptr], transposed_interval) + chord[root_ptr:]
    else:  # no flat or sharp, which means only the first element is the root
        transposed_result = transpose(chord[0], transposed_interval) + chord[1:]
    return transposed_result


def format_sequence_data(inputdim, outputdim, batchsize, x, y):
    """
    Fit the dataset with the size of the batch
    :param inputdim:
    :param outputdim:
    :param batchsize:
    :param x:
    :param y:
    :return:
    """
    yy = [0] * outputdim
    yy[-1] = 1
    while (x.shape[0] % batchsize != 0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    return x, y


def get_predict_file_name(input, data_id, augmentation, bach='Y'):
    filename = []
    num_salami_slices = []
    for id, fn in enumerate(os.listdir(input)):
        if fn.find('KB') != -1:
            if bach == 'Y':
                p = re.compile(r'\d{3}')  # find 3 digit in the file name
                id_id = p.findall(fn)
            elif bach == 'FB':
                p = re.compile(r'\d{1,3}[ab]*')
                id_id = p.findall(fn)
                if len(id_id) == 2:
                    id_id[0] = id_id[0] + '.' + id_id[1]
            else:
                id_id = []
                id_id.append(os.path.splitext(fn)[0])
            if id_id[0] in data_id or data_id == []:  # if the digit found in the list, add this file

                if (augmentation != 'Y'):  # Don't want data augmentation in 12 keys
                    if fn.find('CKE') != -1 or fn.find('C_oriKE') != -1 or fn.find('aKE') != -1 or fn.find('a_oriKE') != -1:  # only wants key c
                        filename.append(fn)
                else:
                    if fn.find('ori') != -1:
                        filename.append(fn)  # only include files that are in the original key
    #filename.sort()
    # filename2 = []  # making sure the sequence is the same of data_id!
    # for i, each_id in enumerate(data_id):
    #     for j, each_file in enumerate(filename):
    #         if each_id in each_file:
    #             filename2.append(each_file)

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
            total += (val * 2 ** index)
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
        print('progress:', i, '/', arr.shape[0])
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


def add_FB_PC(FB, pitchclass, FB_index,chord_tone, i):
    """
    Modular function to add FB pitch class
    :param FB:
    :param pitchclass:
    :param FB_index:
    :param chord_tone:
    :param i:
    :return:
    """
    FB.append(pitchclass[i])
    FB_index.append(i)
    chord_tone[i] = 0
    return FB, FB_index, chord_tone


def get_FB_and_FB_PC(x, y, sChords, j, outputtype, s, key, this_pitch_list, this_pitch_class_list, previous_bass, previous_FB_PC, previous_NCT_sign_FB, RB_reasons, concert_pitch, chordify_voice, type=''):
    """
    'Type' specifies if I want to strip away figures using RB approach
    :param x:
    :param y:
    :param thisChord:
    :param outputtype:
    :param s:
    :return:
    """
    thisChord = sChords.recurse().getElementsByClass('Chord')[j]
    # if thisChord.measureNumber == 5:
    #     print('debug')
    five_three_six_four = 0  # track 53-64 motion
    if j > 0:
        previousChord = sChords.recurse().getElementsByClass('Chord')[j - 1]
        previous_pitch_class_four_voice, previous_pitch_four_voice = get_pitch_class_for_four_voice(previousChord, s)
    else:
        previous_pitch_class_four_voice = []
    if type == 'RB':
        this_pitch_class_list_only_CT, NCT_sign = determine_NCT(sChords, j, s, this_pitch_list, this_pitch_class_list, previous_NCT_sign_FB, concert_pitch, chordify_voice)
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    chord_tone = list(x)
    chord_tone = [int(round(x)) for x in chord_tone]
    actual_FB = []
    if list(y) == [0] * 12:
        # thisChord.addLyric(' ')
        # thisChord.addLyric(' ')  # to align
        return actual_FB, []
    FB = []
    FB_index = []
    bass = get_bass_note(thisChord, this_pitch_list, this_pitch_class_list, 'Y')
    if outputtype.find('_pitch_class') != -1:
        for i, item in enumerate(y):
            if int(item) == 1:
                if bass.pitch.pitchClass == i and bass.pitch.pitchClass not in this_pitch_class_list[:-1]:
                    continue  # Do not need to output bass as FB pitch class, if this PC only appear in bass
                if type == '':  # only CT will be left if RB
                    FB, FB_index, chord_tone = add_FB_PC(FB, pitchclass, FB_index,chord_tone, i)
                elif type == 'RB':
                    if this_pitch_class_list_only_CT[-1] == -2: # if the bass is NCT, then predict no FB!
                        RB_reasons= ['NCT bass'] * len(this_pitch_class_list)
                        break
                    if thisChord.beat % 1 != 0 and thisChord.duration.quarterLength <= 0.25:  # if the current slice contains 16th notes off beat, prediction nothing too!
                        RB_reasons = ['16th (or shorter) note slice ignored'] * len(this_pitch_class_list)
                        break
                    if i in this_pitch_class_list_only_CT:  # if this one is a CT, output as FB in RB approach
                        if previous_bass != -1:
                            if bass.pitch.pitchClass == previous_bass.pitch.pitchClass \
                                    and (pitchclass[i] in previous_FB_PC or i in previous_pitch_class_four_voice):
                                # consider the fact that the same bass can have more than 2 slices with the same PC, none of them should be labelled
                                # another rule: bass remaining the same, and the PC in the current slices does
                                # not need to be labelled if already appeared in the previous slice
                                if bass.pitch.pitchClass not in this_pitch_class_list[:-1]:
                                    # we also need to deal with other voices doubling the bass (e.g., 9-8)
                                    # in this case, we need to keep the 8
                                    # this will leave 8 present even though there is no 9-8 and 8 is found from the previous slice, but this does not affect any functionality
                                    # we need to eliminate 853 since they largely can all be implied anyways
                                    aInterval = interval.Interval(noteStart=bass,
                                                                  noteEnd=this_pitch_list[
                                                                      this_pitch_class_list_only_CT.index(i)])
                                    FB_desired = str(int(aInterval.name[1:]) % 7)
                                    if FB_desired not in ['1', '3', '5']:
                                        print('no need to put this FB PC')
                                        RB_reasons[this_pitch_class_list_only_CT.index(i)] = 'FB already labeled'
                                elif i == bass.pitch.pitchClass:  # only add 8 in this case, since there is a doubling of bass PC
                                    FB, FB_index, chord_tone = add_FB_PC(FB, pitchclass, FB_index, chord_tone, i)
                            else:
                                FB, FB_index, chord_tone = add_FB_PC(FB, pitchclass, FB_index,chord_tone, i)
                        else:
                            FB, FB_index, chord_tone = add_FB_PC(FB, pitchclass, FB_index,chord_tone, i)
                    elif i in this_pitch_class_list:
                        if NCT_sign[this_pitch_class_list.index(i)] == 'NCT_suspension':  # still add this figure since it is SUS!
                            FB, FB_index, chord_tone = add_FB_PC(FB, pitchclass, FB_index, chord_tone, i)

                        # RB_reasons[this_pitch_class_list.index(i)] = 'NCT upper voices: ' + NCT_sign[this_pitch_class_list.index(i)]
                        RB_reasons[this_pitch_class_list.index(i)] = 'NCT upper voices'


        # if FB != []:
        #     thisChord.addLyric(FB)  # output the PC indicated by FB
        # else:
        #     thisChord.addLyric(' ')
        # Output FB to the file too
        pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
        bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')
        for i, sonority in enumerate(pitch_four_voice):
            if hasattr(sonority, 'pitch'):
                if any(sonority.pitch.midi % 12 == each_FB_ptr for each_FB_ptr in FB_index):
                    aInterval = interval.Interval(noteStart=bass, noteEnd=sonority)
                    if int(aInterval.name[1:]) % 7 == 0:
                        FB_desired = '7'
                    elif '9' == aInterval.name[1] or '8' == aInterval.name[1]:
                        FB_desired = aInterval.name[1:]
                    else:
                        FB_desired = str(int(aInterval.name[1:]) % 7)
                    if FB_desired == '1':
                        if i != len(pitch_four_voice) - 1:  # only non-bass can be translated into 8
                            FB_desired = '8'
                        else:
                            continue  # do not need to add bass as FB result!
                    # Sometimes it can share the same pitch class, and in this case,
                    # it will give two identical FB, in this case, we only need one

                    if sonority.pitch.accidental is not None:
                        if not any(sonority.pitch.pitchClass == each_scale.pitchClass for each_scale in key.pitches):
                            if FB_desired == '3':
                                actual_FB = add_FB_result(actual_FB,
                                                                     sonority.pitch.accidental.unicode)
                            elif i != len(pitch_four_voice) - 1:
                                if not (aInterval.name[0] == 'P' and FB_desired == '8'):
                                    actual_FB = add_FB_result(actual_FB,
                                                                         sonority.pitch.accidental.unicode + FB_desired)
                                else:
                                    actual_FB = add_FB_result(actual_FB, FB_desired)  # the exception is where continuo and bass are both raised, we need to output 8 not #8!

                        else:
                            actual_FB = add_FB_result(actual_FB,
                                                                 FB_desired)
                    else:
                        actual_FB = add_FB_result(actual_FB,
                                                             FB_desired)
        if type != 'RB':
            return actual_FB, FB
        else:
            return actual_FB, FB, RB_reasons, NCT_sign


def add_FB_result(actual_FB, result):
    """
    Modular function to add the FB annotation. If already exist, skip this part which will add a duplicated one, since
    there can be situations where multiple voices share the same pitch class
    :param thisChord:
    :param actual_FB:
    :return:
    """
    if result not in actual_FB:
        # thisChord.addLyric(result)
        actual_FB.append(result)
    return actual_FB


def add_FB_result_to_score(thisChord, actual_FB):
    for result in actual_FB:
        thisChord.addLyric(result)
    return thisChord


def output_NCT_to_XML(x, y, thisChord, outputtype):
    """
    Translate 4-bit nct encoding and map the pitch classes and output the result into XML
    If you want to predict_chord, set this parameter to 'Y'
    :param x:
    :param gt:
    :param f_all:
    :param thisChord:
    :return:
    """
    yyptr = -1
    nonchordpitchclassptr = [-1] * 4
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    chord_tone = list(x)
    chord_tone = [int(round(x)) for x in chord_tone]
    # https://stackoverflow.com/questions/35651470/rounding-a-list-of-floats-into-integers-in-python
    #
    if outputtype.find('_pitch_class') == -1:
        for i in range(len(x)):
            if (x[i] == 1):  # non-chord tone
                yyptr += 1
                if (y[yyptr] == 1):
                    nonchordpitchclassptr[yyptr] = i % 12
        if nonchordpitchclassptr != [-1] * 4:
            nct = []  # there are NCTs
            for item in nonchordpitchclassptr:
                if (item != -1):
                    nct.append(pitchclass[item])
                    if len(chord_tone) == 48:
                        for i in range(4):  # Go through each voice and set this class all to 0 (NCT)
                            if int(chord_tone[i * 12 + item]) == 1:
                                chord_tone[i * 12 + item] = 0
                    elif len(chord_tone) == 12:
                        chord_tone[item] = 0
                    else:
                        input('I have a chord tone matrix that I do not know how to process')
            thisChord.addLyric(nct)
        else:
            thisChord.addLyric(' ')
        if len(chord_tone) == 48:
            chord_tone_12 = [0] * 12
            for i, item in enumerate(chord_tone):
                if int(item) == 1:
                    chord_tone_12[i % 12] = 1
            return chord_tone_12
        else:
            return chord_tone
    else:
        # in 12-d pitch class, y is already the chord_tone
        nct = []
        for i, item in enumerate(y):
            if int(item) == 1:
                nct.append(pitchclass[i])
                chord_tone[i] = 0
        if nct != []:
            thisChord.addLyric(nct)
        else:
            thisChord.addLyric(' ')
        return chord_tone

def infer_chord_label1(thisChord, chord_tone, chord_tone_list, chord_label_list):
    """
    Record all the chord tones and chord labels predicted by the model, which are used to finalize the un-determined
    chord
    harmony.chordSymbolFigureFromChord is used to convert pitch classes into chord names
    :param thisChord:
    :param chord_tone:
    :param chord_tone_list:
    :param chord_label_list:
    :return:
    """
    chord_pitch = []
    chord_pitch_class_ID = []
    if int(chord_tone[thisChord.bass().pitchClass]) == 1:  # If bass is a chord tone
        chord_pitch.append(thisChord.pitchNames[thisChord.pitchClasses.index(thisChord.bass().pitchClass)])
        chord_pitch_class_ID.append(thisChord.bass().pitchClass)
        # add the bass note first
    for i, item in enumerate(chord_tone):
        if item == 1:
            if i != thisChord.bass().pitchClass:  # bass note has been added
                if i in thisChord.pitchClasses:
                    chord_pitch.append(thisChord.pitchNames[thisChord.pitchClasses.index(i)])
                    chord_pitch_class_ID.append(i)
    chord_label = chord.Chord(chord_pitch)  # what's wrong with this damn function??? Why I am giving 369 and then give me a f#o????. Must give actual pitch name
    chord_tone_list.append(chord_pitch_class_ID)
    allowed_chord_quality = ['incomplete major-seventh chord', 'major seventh chord',
                             'incomplete minor-seventh chord', 'minor seventh chord',
                             'incomplete half-diminished seventh chord', 'half-diminished seventh chord',
                             'diminished seventh chord',
                             'incomplete dominant-seventh chord', 'dominant seventh chord',
                             'major triad',
                             'minor triad',
                             'diminished triad']
    if chord_tone != [0] * len(chord_tone) and chord_label.pitchClasses != []:  # there must be a slice having at least one chord tone
        if any(each in chord_label.pitchedCommonName for each in allowed_chord_quality):
            # This is the chord we can output directly
            # https://python-forum.io/Thread-Ho-to-check-if-string-contains-substring-from-list
            # thisChord.addLyric(chord_label.pitchedCommonName)
            if harmony.chordSymbolFigureFromChord(chord_label).find('Identified') != -1:  # harmony.chordSymbolFigureFromChord cannot convert pitch classes into chord name sometimes, and the examples are below
                #print('debug')
                if chord_label.pitchedCommonName.find('-diminished triad') != -1: # chord_label.pitchedCommonName is another version of the chord name, but usually I cannot use it to get harmony.ChordSymbol to get pitch classes, so I translate these cases which could be processed by harmony.ChordSymbol later on
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-diminished triad', 'o')) # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete half-diminished seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-incomplete half-diminished seventh chord', '/o7')) # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete minor-seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-incomplete minor-seventh chord', 'm7')) # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete major-seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-incomplete major-seventh chord', 'M7')) # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete dominant-seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-incomplete dominant-seventh chord', '7')) # translate to support
                elif chord_label.pitchedCommonName.find('-major triad') != -1: #(e.g., E--major triad) in  279 slice 33
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-major triad', '')) # translate to support
                elif chord_label.pitchedCommonName.find('-dominant seventh chord') != -1: #(e.g., E--major triad) in  279 slice 33
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-dominant seventh chord', '7')) # translate to support
                elif chord_label.pitchedCommonName.find('-half-diminished seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-half-diminished seventh chord', '/o7')) # translate to support
                elif chord_label.pitchedCommonName.find('-minor-seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-minor-seventh chord', 'm7')) # translate to support
                elif chord_label.pitchedCommonName.find('-major-seventh chord') != -1:
                    chord_label_list.append(chord_label.pitchedCommonName.replace('-major-seventh chord', 'M7')) # translate to support
                else:
                    chord_label_list.append(chord_label.pitchedCommonName)  # Just in case the function cannot accept any names (e.g., E--major triad)
            else:
                if chord_label.pitchedCommonName.find('-incomplete dominant-seventh chord') != -1: # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                    chord_label_list.append(
                        re.sub(r'/[A-Ga-g][b#-]*', '', chord_label.pitchedCommonName.replace('-incomplete dominant-seventh chord', '7')))  # remove 'add' part
                elif chord_label.pitchedCommonName.find('-incomplete major-seventh chord') != -1: # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                    chord_label_list.append(
                        re.sub(r'/[A-Ga-g][b#-]*', '', chord_label.pitchedCommonName.replace('-incomplete major-seventh chord', 'M7')))  # remove 'add' part
                elif harmony.chordSymbolFigureFromChord(chord_label).find('add') != -1: # contains "add" which does not work for harmony.ChordSymbol, at 095
                    chord_label_list.append(
                        re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[:harmony.chordSymbolFigureFromChord(chord_label).find('add')]))  # remove 'add' part
                # elif harmony.chordSymbolFigureFromChord(chord_label).find('power') != -1: # assume power alone as major triad
                #     chord_label_list.append(
                #         re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[
                #                                        :harmony.chordSymbolFigureFromChord(chord_label).find(
                #                                            'power')]))  # remove 'add' part
                elif harmony.chordSymbolFigureFromChord(chord_label).find('dim') != -1:
                    chord_label_list.append(
                        re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label).replace('dim','o')))
                else:
                    chord_label_list.append(re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label))) # remove inversions, notice that half diminished also has /!
                # the line above is the most cases, where harmony.chordSymbolFigureFromChord can give a chord name for the pitch classes, and Bdim is generated by this!
        else:  # undetermined chord, but we want to keep a few cases of 2 note pitch classes
            # thisChord.addLyric('un-determined')
            if chord_label.pitchedCommonName.find('-interval class 5') != -1: # p5 and missing 5th (major third will be considered as major triads)
                chord_label_list.append(chord_label.pitchedCommonName)
            elif chord_label.pitchedCommonName.find('-interval class 4') != -1: # p5 and missing 5th (major third will be considered as major triads)
                chord_label_list.append(chord_label.pitchedCommonName)
            elif chord_label.pitchedCommonName.find('-interval class 3') != -1: # p5 and missing 5th (major third will be considered as major triads)
                chord_label_list.append(chord_label.pitchedCommonName) # minor third missing 5th will be considered as minor triads
            elif chord_label.pitchedCommonName.find('-tritone') != -1: # tritone is converted into diminished chord
                chord_label_list.append(chord_label.pitchedCommonName[:chord_label.pitchedCommonName.find('-tritone')] + 'o')
            else:
                chord_label_list.append('un-determined')
    else:  # no chord tone, this slice is undetermined as well
        # thisChord.addLyric('un-determined')
        chord_label_list.append('un-determined')
    return chord_tone_list, chord_label_list


def infer_chord_label2(j, thisChord, chord_label_list, chord_tone_list):
    """
    Compare the preceding and following chord labels and the one sharing the most chord tone with the current one will
    be considered as the final chord.
    :param j:
    :param thisChord:
    :param chord_label_list:
    :param chord_tone_list:
    :return:
    """
    # TODO: this part of code can be further factorized
    if chord_label_list[-1].find('interval') != -1:  # change the last element first
        if chord_label_list[-1].find(
                '-interval class 5') != -1:  # p5 and missing 5th (major third will be considered as major triads)
            chord_label_list[-1] = chord_label_list[-1].replace('-interval class 5', '')
        elif chord_label_list[-1].find(
                '-interval class 4') != -1:  # p5 and missing 5th (major third will be considered as major triads)
            chord_label_list[-1] = chord_label_list[-1].replace('-interval class 4', '')
        elif chord_label_list[-1].find(
                '-interval class 3') != -1:  # m3 and missing 5th (minor third will be considered as minor triads)
            chord_label_list[-1] = chord_label_list[-1].replace('-interval class 3', '') + 'm'
    if j == 0: # this is the first slice of the song, only look after
        # TODO: always be careful about the first and last slice. Use -1 carefully!
        for jj, itemitem in enumerate(chord_label_list[j + 1:]):
            if itemitem != 'un-determined' and itemitem.find('interval') == -1:  # Find the next real chord
                break
        jj = jj + j + 1
        if chord_label_list[j] == 'un-determined':
            chord_label_list[j] = chord_label_list[jj] # You dont have a choice
            # TODO: this part can get tricky if there are many undetermined slices in the beginning, and we have to look after where slices are not fully processed
        elif chord_label_list[j].find('interval') != -1:
            common_tone2 = list(
                set(chord_tone_list[j]).intersection(harmony.ChordSymbol(chord_label_list[jj]).pitchClasses))
            if len(common_tone2) == len(chord_tone_list[j]): # if the previous slice contains the current slice, use the previous chord
                chord_label_list[j] = chord_label_list[jj]
            else:
                if chord_label_list[j].find('-interval class 5') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 5', '')
                elif chord_label_list[j].find('-interval class 4') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 4', '')
                elif chord_label_list[j].find('-interval class 3') != -1: # m3 and missing 5th (minor third will be considered as minor triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 3', '') + 'm'
    elif j < len(chord_tone_list) - 1:
        for jj, itemitem in enumerate(chord_label_list[j + 1:]):
            if itemitem != 'un-determined' and itemitem.find('interval') == -1:  # Find the next real chord
                break
        jj = jj + j + 1
        #print('j-1', j-1, 'chord_label_list[j - 1]', chord_label_list[j - 1])
        common_tone1 = list(set(chord_tone_list[j]).intersection(harmony.ChordSymbol(chord_label_list[j - 1]).pitchClasses)) # compare the current chord tone with the ones from adjacent chord labels (not the identified tones)
        #print('jj', jj, 'chord_label_list[jj]', chord_label_list[jj])
        #print('j', j, 'chord_label_list[j]', chord_label_list[j])
        if chord_label_list[jj] != 'un-determined':  # it is possible that the last slice of the chorale is undetermined, see 187
            common_tone2 = list(set(chord_tone_list[j]).intersection(harmony.ChordSymbol(chord_label_list[jj]).pitchClasses))
        else:
            common_tone2 = [] # we don't want any chance of the current slice to be 'un-determined'
        if chord_label_list[j] == 'un-determined':
            if len(common_tone1) == len(common_tone2): # if sharing the same number of pcs, choose the chord label whose root is the bass of the slice
                slice_bass = thisChord.bass().pitchClass
                chord_root_1 = harmony.ChordSymbol(chord_label_list[j - 1])._cache['root'].pitchClass
                chord_root_2 = harmony.ChordSymbol(chord_label_list[jj])._cache['root'].pitchClass
                if chord_root_1 == slice_bass:
                    chord_label_list[j] = chord_label_list[j - 1]
                elif chord_root_2 == slice_bass:
                    chord_label_list[j] = chord_label_list[jj]
                else:
                    chord_label_list[j] = chord_label_list[j - 1]
            elif len(common_tone1) > len(common_tone2):
                chord_label_list[j] = chord_label_list[j - 1]
            elif len(common_tone1) < len(common_tone2):
                chord_label_list[j] = chord_label_list[jj]
        elif chord_label_list[j].find('interval') != -1:
            if len(common_tone1) == len(chord_tone_list[j]): # if the previous slice contains the current slice, use the previous chord
                chord_label_list[j] = chord_label_list[j - 1]
            elif len(common_tone2) == len(chord_tone_list[j]): # if the following slice contains the current slice, use the following chord
                chord_label_list[j] = chord_label_list[jj]
            else:
                if chord_label_list[j].find('-interval class 5') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 5', '')
                elif chord_label_list[j].find('-interval class 4') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 4', '')
                elif chord_label_list[j].find('-interval class 3') != -1: # m3 and missing 5th (minor third will be considered as minor triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 3', '') + 'm'
    else: # this is the last slice of the song, only look back
        if chord_label_list[j] == 'un-determined':
            chord_label_list[j] = chord_label_list[j - 1] # only can be changed as the last label
        elif chord_label_list[j].find('interval') != -1:
            common_tone1 = list(
                set(chord_tone_list[j]).intersection(harmony.ChordSymbol(chord_label_list[j - 1]).pitchClasses))
            if len(common_tone1) == len(chord_tone_list[j]): # if the previous slice contains the current slice, use the previous chord
                chord_label_list[j] = chord_label_list[j - 1]
            else:
                if chord_label_list[j].find('-interval class 5') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 5', '')
                elif chord_label_list[j].find('-interval class 4') != -1: # p5 and missing 5th (major third will be considered as major triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 4', '')
                elif chord_label_list[j].find('-interval class 3') != -1: # m3 and missing 5th (minor third will be considered as minor triads)
                    chord_label_list[j] = chord_label_list[j].replace('-interval class 3', '') + 'm'


def infer_chord_label3(j, thisChord, chord_label_list, chord_tone_list):
    # replace chord with adjacent chords which fully contain the chord tones
    if j < len(chord_tone_list) - 1 and j > 0:
        for jj, itemitem in enumerate(chord_label_list[j + 1:]):
            if itemitem != 'un-determined' and itemitem.find('interval') == -1:  # Find the next real chord
                break
        jj = jj + j + 1
        common_tone1 = list(
            set(harmony.ChordSymbol(chord_label_list[j]).pitchClasses).intersection(
                harmony.ChordSymbol(chord_label_list[j - 1]).pitchClasses))
        if chord_label_list[jj] == 'un-determined': # Edge case: 187, the last slice is un-determined
            common_tone2 = []
        else:
            common_tone2 = list(
            set(harmony.ChordSymbol(chord_label_list[j]).pitchClasses).intersection(
                harmony.ChordSymbol(chord_label_list[jj]).pitchClasses))
        if len(common_tone1) == len(harmony.ChordSymbol(chord_label_list[j]).pitchClasses) and len(
                harmony.ChordSymbol(chord_label_list[j - 1]).pitchClasses) > len(
            harmony.ChordSymbol(chord_label_list[j]).pitchClasses):
            chord_label_list[j] = chord_label_list[j - 1]
        if len(common_tone2) == len(harmony.ChordSymbol(chord_label_list[j]).pitchClasses) and len(
                harmony.ChordSymbol(chord_label_list[jj]).pitchClasses) > len(
            harmony.ChordSymbol(chord_label_list[j]).pitchClasses):
            chord_label_list[j] = chord_label_list[jj]


def create_3D_data(x, timestep):
    """
    generate 3D data for RNN like network to train based on 2D data.
    For the slice ID that is smaller than timestep, append 0 vector
    :param x:
    :param timestep:
    :return:
    """
    xx = []
    blank = [0] * x.shape[-1]
    for i in range(x.shape[0]):
        beginning = []
        if i < timestep - 1:
            for j in range(timestep - i - 1):
                beginning.append(blank)
            beginning = np.vstack((beginning, x[0:i + 1]))
            xx.append(beginning)
        else:
            xx.append(x[i - timestep + 1:i + 1,])
    xx = np.array(xx)
    return xx


def generate_ML_matrix(augmentation, portion, id, model, windowsize, ts, path, sign='N'):
    """

    :param id: Id for training, validating and testing files
    :param model: Model ID. If RNN variant used a different windowing schema
    :param windowsize: For non-RNN variant
    :param ts: For RNN variant
    :param path: The folder for the encoding files
    :param sign: A sign indicating whether the pitch_class_only encoding is needed
    :return:
    """
    counter = 0
    #encoding_all = []
    fn_all = [] # Unify the order
    for fn in os.listdir(path):
        if 'N' in sign: # eliminate pitch class only encoding
            if fn.find('_pitch_class') != -1 or fn.find('_chord_tone') != -1:
                continue
        elif 'Y' in sign : # only want pitch class only encoding
            if fn.find('_pitch_class') == -1:
                continue
        elif sign == 'C': # only want chord tone as input to train the chord inferral algorithm
            if fn.find('_chord_tone') == -1:
                continue
        if augmentation == 'N':
            if fn.find('CKE') == -1 and fn.find('C_oriKE') == -1 and fn.find('aKE') == -1 and fn.find('a_oriKE') == -1: # we cannot find key of c, skip
                continue
        elif portion == 'valid' or portion == 'test': # we want original key on valid and test set when augmenting
            if fn.find('_ori') == -1:
                continue
        if 'FB' not in sign:
            p = re.compile(r'\d{3}')  # find 3 digit in the file name
        else:
            p = re.compile(r'\d{1,3}[ab]*')
        id_id = p.findall(fn)
        if 'FB' in sign and len(id_id) == 2:
            id_id[0] = id_id[0] + '.' + id_id[1]
        if id_id[0] == '137':
            print('debug')
        if id_id[0] in id:
            fn_all.append(fn)
    print(fn_all)
    fn_all.sort()
    print(fn_all)
    for fn in fn_all:
        encoding = np.loadtxt(os.path.join(path, fn))
        if path.find('_x_') != -1:  # we need to add windows
            if model.find('SVM') != -1 or model.find('DNN') != -1 or model.find('DT') != -1:
                encoding_window = adding_window_one_hot(encoding, windowsize)
            elif ts != 0:
                encoding_window = create_3D_data(encoding, ts)
            else:
                encoding_window = list(encoding) # If ts is 0, we dont do anything
            if counter == 0:
                encoding_all = list(encoding_window)
                encoding_all = np.array(encoding_all)
            else:
                encoding_all = np.concatenate((encoding_all, encoding_window))
        else:
            if counter == 0:
                encoding_all = list(encoding)
                encoding_all = np.array(encoding_all)
            else:
                encoding_all = np.concatenate((encoding_all, encoding))
        counter += 1
    print(portion, 'finished')
    return encoding_all, fn_all


def train_ML_model(modelID, HIDDEN_NODE, layer, timestep, outputtype, patience, sign, FOLDER_NAME, MODEL_NAME, batch_size, epochs, csv_logger, train_xx, train_yy, valid_xx, valid_yy):
    INPUT_DIM = train_xx.shape[1]
    OUTPUT_DIM = train_yy.shape[1]
    print('train_xx shape:', train_xx.shape)
    print('train_yy shape:', train_yy.shape)
    print('valid_xx shape:', valid_xx.shape)
    print('valid_yy shape:', valid_yy.shape)
    if modelID.find('SVM') == -1 and modelID.find('DT') == -1:
        model = Sequential()
        # model.add(Embedding(36, 256, input_length=batch))
        if modelID.find('DNN') != -1:
            model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh', input_dim=INPUT_DIM))
            model.add(Dropout(0.2))
            for i in range(layer - 1):
                model.add(Dense(HIDDEN_NODE, init='uniform', activation='tanh'))
                model.add(Dropout(0.2))
        else:
            if modelID.find('BLSTM') != -1:
                model.add(Bidirectional(
                    LSTM(return_sequences=True, dropout=0.2, input_shape=(timestep, INPUT_DIM),
                         units=HIDDEN_NODE,
                         )))
                for i in range(layer - 1):
                    model.add(
                        Bidirectional(
                            LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2)))
                if modelID.find('attention') != -1:
                    model.add(
                        Bidirectional(
                            LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2)))
                    model.add(SeqSelfAttention(attention_activation='sigmoid'))
                else:
                    model.add(
                        Bidirectional(
                            LSTM(units=HIDDEN_NODE, dropout=0.2)))
            elif modelID.find('RNN') != -1:
                model.add(
                    SimpleRNN(input_shape=(timestep, INPUT_DIM), units=HIDDEN_NODE, return_sequences=True, dropout=0.2))
                for i in range(layer - 2):
                    model.add(
                        SimpleRNN(units=HIDDEN_NODE, return_sequences=True, dropout=0.2))
                if modelID.find('attention') != -1:
                    model.add(
                        SimpleRNN(units=HIDDEN_NODE, return_sequences=True, dropout=0.2))
                    model.add(SeqSelfAttention(attention_activation='sigmoid'))
                else:
                    model.add(
                        SimpleRNN(units=HIDDEN_NODE, dropout=0.2))
            elif modelID.find('LSTM') != -1:
                model.add(
                    LSTM(return_sequences=True, dropout=0.2, input_shape=(timestep, INPUT_DIM),
                         units=HIDDEN_NODE))  # , input_shape=train_xx.shape)
                for i in range(layer - 2):
                    model.add(LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2))
                if modelID.find('attention') != -1:
                    model.add(
                        LSTM(units=HIDDEN_NODE, return_sequences=True, dropout=0.2))
                    model.add(SeqSelfAttention(attention_activation='sigmoid'))
                else:
                    model.add(
                        LSTM(units=HIDDEN_NODE, dropout=0.2))
        model.add(Dense(OUTPUT_DIM))
        nadam = optimizers.nadam(lr=0.001)
        if outputtype.find("NCT") != -1:

            if MODEL_NAME.find('chord_tone') == -1: # this is just normal NCT training
                model.add(Activation('sigmoid'))
                model.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['binary_accuracy'])
            else: # training chord inferral model for NCT
                model.add(Activation('softmax'))
                model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
        elif outputtype == "CL":
            model.add(Activation('softmax'))
            model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)  # set up early stopping
        print("Train...")
        checkpointer = ModelCheckpoint(filepath=os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5",
                                       verbose=1, save_best_only=True, monitor='val_loss')
        tbCallBack = TensorBoard(log_dir=os.path.join('.', 'ML_result', sign, FOLDER_NAME), histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)
        hist = model.fit(train_xx, train_yy, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                         validation_data=(valid_xx, valid_yy),
                         callbacks=[early_stopping, checkpointer, csv_logger, tbCallBack])
        return model
    elif modelID == "SVM":
        if outputtype.find("CL") != -1 or MODEL_NAME.find('chord_tone') != -1:
            model = SVC(verbose=True, kernel='linear')
            train_yy_int = np.asarray(onehot_decode(train_yy))
            valid_yy_int = np.asarray(onehot_decode(valid_yy))
            train_xx_SVM = np.vstack((train_xx, valid_xx))
            train_yy_int_SVM = np.concatenate((train_yy_int, valid_yy_int))
            print('new training set', train_xx_SVM.shape, train_yy_int_SVM.shape)
            model.fit(train_xx_SVM, train_yy_int_SVM)
            return model
        elif outputtype.find("NCT") != -1:  # we need to do multilabel classification
            model = OneVsRestClassifier(SVC(verbose=True, kernel='linear'))
            train_xx_SVM = np.vstack((train_xx, valid_xx))
            train_yy_SVM = np.concatenate((train_yy, valid_yy))
            print('new training set', train_xx_SVM.shape, train_yy_SVM.shape)
            model.fit(train_xx_SVM, train_yy_SVM)
            return model
    elif modelID == 'DT':  # we want to learn the rules and visualize the rules
        if outputtype.find("NCT") != -1:
            model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            train_xx_DT = np.vstack((train_xx, valid_xx))
            train_yy_DT = np.concatenate((train_yy, valid_yy))
            print('new training set', train_xx_DT.shape, train_yy_DT.shape)
            model.fit(train_xx_DT, train_yy_DT)
            # tree_FB = TREE.export_graphviz(model, feature_names= ["bass_" + pc for pc in c2] + c2 + ["real_attack_" + pc for pc in c2] + ['downbeat', 'onbeat', 'offbeat'], filled=True, rounded=True, leaves_parallel=True, out_file=None)
            # graph = graphviz.Source(tree_FB, format="pdf")
            # graph.render(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME), view=True)
            return model
        elif outputtype.find("CL") != -1 or MODEL_NAME.find('chord_tone') != -1:
            input('DT for chord labeling has not been developed!')


def unify_GTChord_and_inferred_chord(name):
    if name.find('M')!= -1 and name.find('M7') == -1:
        name = name.replace('M', '')
    if name.find('maj7') != -1:
        name = name.replace('maj7', 'M7')
    return name[0] + name[1:]


def determine_potential_sus(fig_a, fig_b, FB, previous_FB, previous_bass, bass, ptr, s, sChord):
    # if fig_a not in FB:
    if fig_a not in previous_FB or previous_bass == -1 or previous_bass.name == 'rest' or bass.name == 'rest':
        if fig_b in FB:
            FB.remove(fig_b)
    elif previous_bass.name != 'rest' and bass.name != 'rest':
        pitch_class_four_voice, pitch_four_voice = \
            get_pitch_class_for_four_voice(sChord.recurse().getElementsByClass('Chord')[ptr - 1], s)
        for voice_number, sonority in enumerate(pitch_four_voice):  # Don't use thisChord._notes anymore since that
            # has two problems: (1) note will be collapsed if doubled, (2) it ranks as pitch class, not the actual voice!
            if pitch_class_four_voice[voice_number] != -1:
                aInterval = interval.Interval(noteStart=previous_bass, noteEnd=sonority)
                colllapsed_interval = colllapse_interval(aInterval.name[1:])
                if fig_a == colllapsed_interval or int(fig_a) - int(colllapsed_interval) == 7:  # found the voice that has this figure, also consider 9 vs 2 problem
                    if is_suspension(ptr - 1, 1, s, sChord, voice_number, fig_a) == False:  # TO do this, you need to uncollapse the voices
                        if fig_b in FB:
                            FB.remove(fig_b)
                    else:
                        sChord.recurse().getElementsByClass('Chord')[ptr - 1].style.color = 'pink'
                        # At this point, 2-8 should be 9-8
                        for i, each_FB in enumerate(previous_FB):
                            if each_FB == '2':
                                previous_FB[i] = '9'
                    # note that only 9-8, 6-5, and 4-3 suspensions are labelled in this case
    return previous_FB, FB


def remove_implied_FB(gt_FB, predict_FB, previous_gt_FB, previous_predict_FB, previous_bass, bass, ptr, s, sChord):
    """

    :param gt_FB:
    :param predict_FB:
    :param previous_gt_FB:
    :param previous_predict_FB:
    :param previous_bass:
    :param bass:
    :param ptr:
    :param s:
    :param sChord:
    :return:
    """
    previous_gt_FB, gt_FB = determine_potential_sus('4', '3', gt_FB, previous_gt_FB, previous_bass, bass, ptr, s, sChord)
    previous_predict_FB, predict_FB = determine_potential_sus('4', '3', predict_FB, previous_predict_FB, previous_bass, bass, ptr, s, sChord)
    previous_gt_FB, gt_FB = determine_potential_sus('6', '5', gt_FB, previous_gt_FB, previous_bass, bass, ptr, s, sChord)
    previous_predict_FB, predict_FB = determine_potential_sus('6', '5', predict_FB, previous_predict_FB, previous_bass, bass, ptr, s, sChord)
    if '2' not in previous_gt_FB:
        previous_gt_FB, gt_FB = determine_potential_sus('9', '8', gt_FB, previous_gt_FB, previous_bass, bass, ptr, s, sChord)
    elif '9' not in previous_gt_FB:
        previous_gt_FB, gt_FB = determine_potential_sus('2', '8', gt_FB, previous_gt_FB, previous_bass, bass, ptr, s, sChord)
    if '2' not in previous_predict_FB:
        previous_predict_FB, predict_FB = determine_potential_sus('9', '8', predict_FB, previous_predict_FB,
                                                                  previous_bass, bass, ptr, s, sChord,)
    elif '9' not in previous_predict_FB:
        previous_predict_FB, predict_FB = determine_potential_sus('2', '8', predict_FB, previous_predict_FB,
                                                                  previous_bass, bass, ptr, s, sChord,)
    # TODO: resolve this 9 vs 2 issue!
    # Stop using because of the 8-7 problem
    if '4' in gt_FB and ('2' in gt_FB or '3' in gt_FB) and '7' not in previous_gt_FB:
        if '6' in gt_FB and len(gt_FB) == 3:
            gt_FB.remove('6')
    if '4' in predict_FB and ('2' in predict_FB or '3' in predict_FB) and '7' not in previous_predict_FB:
        if '6' in predict_FB and len(predict_FB) == 3:
            predict_FB.remove('6')
    return previous_gt_FB, gt_FB, previous_predict_FB, predict_FB


def count_correct_slices(predict_FB_PC, gt_FB_PC, gt_FB, predict_FB, correct_num, correct_num_implied, correct_mark_all):
    """
    The modular function that counts the correct slides of (implied) FB
    :param correct_bit:
    :param gt:
    :param correct_num:
    :param correct_num_implied:
    :param thisChord:
    :return:
    """
    for each_figure in ['3', '5', '8']:
        if each_figure not in gt_FB:  # this means these intervals can still be implied since GT does not have it
            if each_figure in predict_FB:
                predict_FB.remove(each_figure)
    if predict_FB_PC == gt_FB_PC:
        correct_num += 1
        correct_num_implied += 1
        correct_mark_all.append('✓')
    elif gt_FB == predict_FB:
        correct_num_implied += 1
        correct_mark_all.append('✓_')
    else:
        correct_mark_all.append(' ')
    return correct_num, correct_num_implied, correct_mark_all


def one_hot_PC_filler(list_number):
    pitchclass = [0] * 12
    for i in list_number:
        pitchclass[i] = 1
    return pitchclass




def output_FB_to_score(FB, FB_PC, sChords, j, correct_mark=''):
    thisChord = sChords.recurse().getElementsByClass('Chord')[j]
    if FB_PC != []:  # output PC first
        thisChord.addLyric(FB_PC)  # output the PC indicated by FB
    else:
        thisChord.addLyric(' ')
    for each_FB in FB:
        thisChord.addLyric(each_FB)
    space_needed = 3 - len(FB)  # align the results
    for i in range(space_needed):
        thisChord.addLyric(' ') # beautify formatting
    if correct_mark != '':
        thisChord.addLyric(correct_mark)





def train_and_predict_FB(layer, nodes, windowsize, portion, modelID, ts, bootstraptime, sign, augmentation,
                                     cv, pitch_class, ratio, input, output, balanced, outputtype,
                                     inputtype, predict, exclude=[]):
    print('Training and testing the machine learning models')
    id_sum = find_id_FB(input, exclude)
    num_of_chorale = len(id_sum)
    train_num = num_of_chorale - int(round((num_of_chorale * (1 - ratio) / 2))) * 2
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
    frame_acc = []
    frame_acc_implied = []
    frame_acc_RB = []
    frame_acc_RB_implied = []
    frame_acc_2 = []
    cvscores_percentage_of_NCT_per_slice = []
    NCT_bass_acc = []
    NCT_bass_percentage = []
    NCT_upper_acc = []
    NCT_upper_percentage = []
    FB_already_labeled_acc = []
    FB_already_labeled_percentage = []
    sixteenth_note_acc = []
    sixteenth_note_percentage = []
    empty_FB_acc = []
    empty_FB_percentage = []
    NCT_bass_err = []
    NCT_upper_err = []
    FB_already_labeled_err = []
    sixteenth_note_err = []
    empty_FB_err = []
    # ML copy of it
    NCT_bass_acc_ML = []
    NCT_bass_percentage_ML = []
    NCT_upper_acc_ML = []
    NCT_upper_percentage_ML = []
    FB_already_labeled_acc_ML = []
    FB_already_labeled_percentage_ML = []
    sixteenth_note_acc_ML = []
    sixteenth_note_percentage_ML = []
    empty_FB_acc_ML = []
    empty_FB_percentage_ML = []
    ML_rule_acc = []
    ML_rule_percentage = []
    NCT_bass_err_ML = []
    NCT_upper_err_ML = []
    FB_already_labeled_err_ML = []
    sixteenth_note_err_ML = []
    empty_FB_err_ML = []
    ML_rule_err = []
    batch_size = 256
    epochs = 500
    if modelID == 'DNN':
        patience = 50
    else:
        patience = 20
    print('Loading data...')
    extension = sign + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21 + '_' + 'training' + str(
        train_num)
    timestep = ts
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                 str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                 + str(timestep) + extension
    print('Loading data...')
    print('Build model...')
    if not os.path.isdir(os.path.join('.', 'ML_result', sign)):
        os.mkdir(os.path.join('.', 'ML_result', sign))
    if not os.path.isdir(os.path.join('.', 'ML_result', sign, MODEL_NAME)):
        os.mkdir(os.path.join('.', 'ML_result', sign, MODEL_NAME))
    cv_log = open(os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+') + 'predict.txt', 'w')
    csv_logger = CSVLogger(os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+') + 'predict_log.csv',
                           append=True, separator=';')
    for times in range(cv):
        if times != 5 :
            continue
        MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                     str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                     + str(timestep) + extension  + '_cv_' + str(times + 1)
        FOLDER_NAME = 'MODEL'
        train_id, valid_id, test_id = get_id(id_sum, num_of_chorale, times)
        # if exclude != []:
        #     train_id.extend(test_id)
        #     test_id = exclude # Swap the test id into the 39 ones
        valid_yy, fileName_fake = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts,
                                      os.path.join('.', 'data_for_ML', sign,
                                                   sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB')
        valid_xx, fileName_fake = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts,
                                      os.path.join('.', 'data_for_ML', sign,
                                                   sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB_N')
        if not (os.path.isfile((os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5"))):
            train_xx, fileName_fake = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                          os.path.join('.', 'data_for_ML', sign,
                                                       sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB_N')
            train_yy, fileName_fake = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                          os.path.join('.', 'data_for_ML', sign,
                                                       sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB')
            train_xx = train_xx[
                       :int(portion * train_xx.shape[0])]  # expose the option of training only on a subset of data
            train_yy = train_yy[:int(portion * train_yy.shape[0])]
            print('training and predicting...')
            model = train_ML_model(modelID, HIDDEN_NODE, layer, timestep, outputtype, patience, sign,
                                   FOLDER_NAME, MODEL_NAME, batch_size, epochs, csv_logger, train_xx, train_yy,
                                   valid_xx,
                                   valid_yy)  # train the machine learning model
        else:
            model = load_model(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5")
        test_xx, fileName = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts,
                                     os.path.join('.', 'data_for_ML', sign,
                                                  sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB_N')
        test_xx_only_pitch, fileName = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts,
                                                os.path.join('.', 'data_for_ML', sign,
                                                             sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                                'FB_Y')
        test_yy, fileName = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts,
                                     os.path.join('.', 'data_for_ML', sign,
                                                  sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'FB')

        predict_y = model.predict(test_xx)  # Predict the probability for each bit of FB sonority
        for i in predict_y:  # regulate the prediction
            for j, item in enumerate(i):
                if (item > 0.5):
                    i[j] = 1
                else:
                    i[j] = 0
        correct_num2 = 0
        for i, item in enumerate(predict_y):
            if np.array_equal(item, test_yy[
                i]):  # https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
                correct_num2 += 1
        frame_acc_2.append(((correct_num2 / predict_y.shape[0]) * 100))
        precision, recall, f1score, accuracy, true_positive, false_positive, false_negative, true_negative = evaluate_f1score(
            model, valid_xx, valid_yy, modelID)
        precision_test, recall_test, f1score_test, accuracy_test, asd, sdf, dfg, fgh = evaluate_f1score(model,
                                                                                                        test_xx,
                                                                                                        test_yy,
                                                                                                        modelID)
        num_of_NCT_slices = 0
        for i, item in enumerate(test_yy):
            if 1 in item:
                num_of_NCT_slices += 1
        cvscores_percentage_of_NCT_per_slice.append((num_of_NCT_slices / test_yy.shape[0]) * 100)
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
        if predict == 'Y':
            # prediction put into files
            for i, each_file in enumerate(fileName):
                fileName[i] = fileName[i][:-3] + 'xml'
            numSalamiSlices = []
            for id, FN in enumerate(fileName):
                length = 0
                s = converter.parse(os.path.join(input, FN))
                sChords = s.chordify()
                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    length += 1
                numSalamiSlices.append(length)
            # fileName, numSalamiSlices = get_predict_file_name(input, test_id, augmentation, 'FB')
            sum = 0
            for i in range(len(numSalamiSlices)):
                sum += numSalamiSlices[i]
            # input(sum)
            # input(predict_y.shape[0])

            length = len(fileName)
            a_counter = 0
            a_counter_correct = 0
            a_counter_correct_implied = 0
            a_counter_correct_RB = 0
            a_counter_correct_RB_implied = 0
            a_all_RB_reasons = []  # RB reason across all pieces
            a_all_ML_reasons = []
            a_gt_FB_all = []  # GT FB across all pieces
            a_predict_FB_RB_all = []
            a_correct_mark_all_RB = []
            a_correct_mark_all_ML = []
            if not os.path.isdir(os.path.join('.', 'predicted_result', sign)):
                os.mkdir(os.path.join('.', 'predicted_result', sign))
            if not os.path.isdir(os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(windowsize + 1))):
                os.mkdir(os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID  + str(windowsize) + '_' + str(windowsize + 1)))
            if times == 0:
                f_all = open(os.path.join('.', 'predicted_result', sign,
                                          outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(
                                              windowsize + 1), 'ALTOGETHER_SUS_ME_3') + str(times) + '.txt',
                             'w')  # create this file to track every type of mistakes
            else:
                f_all = open(os.path.join('.', 'predicted_result', sign,
                                          outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(
                                              windowsize + 1), 'ALTOGETHER_SUS_ME_3') + str(times) + '.txt',
                             'a')  # create this file to track every type of mistakes
            # the sequence of the filename should be the same with test_id!

            for i in range(length):
                print(fileName[i][:-4], file=f_all)
                print(fileName[i])
                # if '13.06' not in fileName[i]:
                #     if '133.06' not in fileName[i]:
                #         continue
                # if '248.46' not in fileName[i]: continue
                num_salami_slice = numSalamiSlices[i]
                correct_num = 0
                correct_num_implied = 0
                correct_num_RB = 0
                correct_num_implied_RB = 0
                gt_FB_all = []
                gt_FB_PC_all = []
                gt_FB_all_implied = []
                predict_FB_all = []
                predict_FB_PC_all = []
                predict_FB_all_implied = []
                predict_FB_RB_all = []
                predict_FB_RB_PC_all = []
                all_RB_reasons = []
                all_ML_reasons = []
                predict_FB_RB_all_implied = []
                correct_mark_all = []
                correct_mark_all_RB = []

                s = converter.parse(os.path.join(input, fileName[i]))  # the source musicXML file
                concert_pitch = contain_concert_pitch(s)
                chordify_voice = contain_chordify_voice(s)
                # s = remove_concert_pitch_voices(s)
                s_no_chordify = converter.parse(os.path.join(input, fileName[i]))  # remove the last voice which will always be the chorify voice
                # s_no_chordify = remove_concert_pitch_voices(s_no_chordify)
                s_no_chordify.remove(s_no_chordify.parts[-1]) # remove the last voice which will always be the chorify voice
                sChords = s_no_chordify.chordify()
                sChords_RB = s_no_chordify.chordify()
                k = s.analyze('AardenEssen')
                previous_gt_FB = []
                previous_predict_FB = []
                previous_predict_FB_RB = []
                previous_predict_FB_RB_PC = []
                previous_bass = -1
                previous_NCT_sign= [''] * 100  # a dummy first one

                for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    # print('measure', thisChord.measureNumber)
                    # if '112.05' in fileName[i] and thisChord.measureNumber == 3:
                    #     print('debug')
                    # if j == 14:
                    #     print('debug')
                    pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
                    NCT_sign = [''] * len(pitch_class_four_voice)
                    bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')
                    RB_reasons = [''] * len(pitch_class_four_voice)  # have reasons for each pitch!
                    ML_reasons = [''] * len(pitch_class_four_voice)
                    gt = test_yy[a_counter]
                    prediction = predict_y[a_counter]
                    correct_bit = 0
                    correct_bit_RB = 0
                    # for ii in range(len(gt)):
                    #     if (gt[ii] == prediction[ii]):  # the label is correct
                    #         correct_bit += 1
                    #     if gt[ii] == one_hot_PC_filler(pitch_class_four_voice[:-1])[ii]:
                    #         correct_bit_RB += 1
                    dimension = test_xx_only_pitch.shape[1]

                    realdimension = int(dimension / (2 * windowsize + 1))
                    if modelID != 'SVM' and modelID != 'DNN' and modelID != 'DT':
                        x = test_xx_only_pitch[a_counter][-1]  # no window if the matrix is 3D
                    else:
                        if windowsize < 0:
                            x = test_xx_only_pitch[a_counter]
                        else:
                            x = test_xx_only_pitch[a_counter][
                                realdimension * windowsize:realdimension * (windowsize + 1)]
                    gt_FB, gt_FB_PC = get_FB_and_FB_PC(x, gt, sChords, j, outputtype, s, k, pitch_four_voice, pitch_class_four_voice, previous_bass, [], [], RB_reasons, concert_pitch, chordify_voice)  # Get the resulting PC and FB
                    if gt_FB == ['2', '6', '4']:
                        print('debug')
                    # if j == 25 and '248.53' in fileName[i]:
                    #     print('debug')
                    predict_FB, predict_FB_PC = get_FB_and_FB_PC(x, prediction, sChords, j, outputtype, s, k, pitch_four_voice, pitch_class_four_voice, previous_bass, [], [], RB_reasons, concert_pitch, chordify_voice)
                    predict_FB_RB, predict_FB_RB_PC, RB_reasons, NCT_sign = get_FB_and_FB_PC(x, one_hot_PC_filler(pitch_class_four_voice), sChords_RB, j, outputtype, s, k, pitch_four_voice, pitch_class_four_voice, previous_bass, previous_predict_FB_RB_PC, previous_NCT_sign, RB_reasons, concert_pitch, chordify_voice, 'RB')
                    if predict_FB_RB == ['5', '8', '4'] and '248.53' in fileName[i]:
                        print('debug')
                    if predict_FB_RB == ['8', '7', '3'] and '30.06' in fileName[i]:
                        print('debug')
                    previous_gt_FB, gt_FB_implied, previous_predict_FB, predict_FB_implied= remove_implied_FB(list(gt_FB), list(predict_FB), previous_gt_FB, previous_predict_FB, previous_bass, bass, j, s_no_chordify, sChords)  # here, all implied intervals are removed, but not printed to score. Suspension is not considered since it cannot be implied
                    #print('previous_predict_FB_RB before', previous_predict_FB_RB)
                    previous_fake_gt_FB, fake_gt_FB_implied, previous_predict_FB_RB, predict_FB_RB_implied = remove_implied_FB(list(gt_FB), list(predict_FB_RB), previous_gt_FB, previous_predict_FB_RB, previous_bass, bass, j, s_no_chordify, sChords_RB)
                    #print('previous_predict_FB_RB after', previous_predict_FB_RB)
                    # Save results

                    gt_FB_all.append(gt_FB)
                    gt_FB_PC_all.append(gt_FB_PC)
                    gt_FB_all_implied.append(gt_FB_implied)
                    predict_FB_all.append(predict_FB)
                    predict_FB_PC_all.append(predict_FB_PC)
                    predict_FB_all_implied.append(predict_FB_implied)
                    predict_FB_RB_all.append(predict_FB_RB)
                    predict_FB_RB_PC_all.append(predict_FB_RB_PC)
                    predict_FB_RB_all_implied.append(predict_FB_RB_implied)
                    all_RB_reasons.append(RB_reasons)
                    a_all_RB_reasons.append(RB_reasons)
                    # need to add ML reasons, same if the implied FB is the same, specify "ML rules" if different
                    if predict_FB_implied == predict_FB_RB_implied:
                        all_ML_reasons.append(RB_reasons)
                        a_all_ML_reasons.append(RB_reasons)
                    else:
                        ML_reasons = ['ML rules'] * len(pitch_class_four_voice)
                        all_ML_reasons.append(ML_reasons)
                        a_all_ML_reasons.append(ML_reasons)
                    if j != 0:
                        gt_FB_all[j - 1] = previous_gt_FB
                        predict_FB_all[j - 1] = previous_predict_FB
                        predict_FB_RB_all[j - 1] = previous_predict_FB_RB
                    previous_gt_FB = gt_FB
                    previous_predict_FB = predict_FB
                    previous_predict_FB_RB = predict_FB_RB
                    previous_bass = bass
                    previous_predict_FB_RB_PC = predict_FB_RB_PC
                    previous_NCT_sign = NCT_sign
                    correct_num, correct_num_implied, correct_mark_all = count_correct_slices(predict_FB_PC, gt_FB_PC, gt_FB_implied, predict_FB_implied, correct_num, correct_num_implied, correct_mark_all)
                    correct_num_RB, correct_num_implied_RB, correct_mark_all_RB = count_correct_slices(predict_FB_RB_PC, gt_FB_PC, gt_FB_implied, predict_FB_RB_implied,
                                                                            correct_num_RB, correct_num_implied_RB, correct_mark_all_RB)
                    # if (correct_bit == len(gt)):
                    #     correct_num += 1
                    #     correct_num_implied += 1
                    #     thisChord.addLyric('✓')
                    # elif gt_FB == predict_FB:
                    #     correct_num_implied += 1
                    #     thisChord.addLyric('✓_')
                    thisChord.closedPosition(forceOctave=4, inPlace=True)
                    sChords_RB.recurse().getElementsByClass('Chord')[j].closedPosition(forceOctave=4, inPlace=True)
                    a_counter += 1
                for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')): # output all the results to score
                    output_FB_to_score(gt_FB_all[j], gt_FB_PC_all[j], sChords, j)
                    output_FB_to_score(predict_FB_all[j], predict_FB_PC_all[j], sChords, j, correct_mark_all[j])
                    output_FB_to_score(predict_FB_RB_all[j], predict_FB_RB_PC_all[j], sChords_RB, j, correct_mark_all_RB[j])
                    if correct_mark_all_RB[j] == '✓_' or correct_mark_all_RB[j] == '✓':
                        continue
                    else:
                        print('Error at m.', thisChord.measureNumber, 'beat', thisChord.beat, 'reason', all_RB_reasons[j], 'GT FB', gt_FB_all[j], 'Pre FB', predict_FB_RB_all[j])
                        # print('Error at m.', thisChord.measureNumber, 'beat', thisChord.beat, 'reason',
                        #       all_RB_reasons[j], 'GT FB', gt_FB_all[j], 'Pre FB', predict_FB_RB_all[j], file=f_all)
                s.insert(0, sChords)
                s.insert(0, sChords_RB)
                a_counter_correct += correct_num
                a_counter_correct_implied += correct_num_implied
                a_counter_correct_RB += correct_num_RB
                a_counter_correct_RB_implied += correct_num_implied_RB
                a_predict_FB_RB_all.append(predict_FB_RB_all)  # add implied ones easy to compare
                a_gt_FB_all.append(gt_FB_all)  # same with above
                a_correct_mark_all_RB.append(correct_mark_all_RB)
                a_correct_mark_all_ML.append(correct_mark_all)
                print(end='\n', file=f_all)


                print('frame accucary: ' + str(correct_num / num_salami_slice), end='\n', file=f_all)
                print('num of correct frame answers: ' + str(correct_num) + ' number of salami slices: ' + str(
                    num_salami_slice),
                      file=f_all)
                print('frame accucary implied: ' + str(correct_num_implied / num_salami_slice), end='\n', file=f_all)
                print('num of correct frame answers implied: ' + str(correct_num_implied) + ' number of salami slices: ' + str(
                    num_salami_slice),
                      file=f_all)
                print('accumulative frame accucary: ' + str(a_counter_correct / a_counter), end='\n', file=f_all)
                print('accumulative frame accucary implied: ' + str(a_counter_correct_implied / a_counter), end='\n', file=f_all)
                print('accumulative frame accucary RB: ' + str(a_counter_correct_RB / a_counter), end='\n', file=f_all)
                print('accumulative frame accucary implied RB: ' + str(a_counter_correct_RB_implied / a_counter), end='\n',
                      file=f_all)
                s.write('musicxml',
                        fp=os.path.join('.', 'predicted_result', sign,
                                        outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(
                                            windowsize + 1), fileName[i][
                                                             :-4]) + '.xml')
            frame_acc.append((a_counter_correct / a_counter) * 100)
            frame_acc_implied.append((a_counter_correct_implied / a_counter) * 100)
            frame_acc_RB.append((a_counter_correct_RB / a_counter) * 100)
            frame_acc_RB_implied.append((a_counter_correct_RB_implied / a_counter) * 100)
            f_all.close()
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
                print('Test f1:', i, f1_test[i], '%',  'Frame acc:', frame_acc[i], 'Frame acc implied:', frame_acc_implied[i], '%', file=cv_log)
                print('Test f1:', i, f1_test[i], '%', 'Frame acc RB:', frame_acc_RB[i], 'Frame acc RB implied:',
                      frame_acc_RB_implied[i], '%', file=cv_log)
            print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
            print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
            print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
            print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
            print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', )
            print('Test % of NCTs per slice:', np.mean(cvscores_percentage_of_NCT_per_slice), '%', '±',
                  np.std(cvscores_percentage_of_NCT_per_slice), '%', file=cv_log)
            print('Test % of NCTs per slice:', np.mean(cvscores_percentage_of_NCT_per_slice), '%', '±',
                  np.std(cvscores_percentage_of_NCT_per_slice), '%')
            print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)
            print('Test frame acc:', np.mean(frame_acc), '%', '±', np.std(frame_acc), '%', file=cv_log)
            print('Test frame acc:', np.mean(frame_acc), '%', '±', np.std(frame_acc), '%')
            print('Test frame acc implied:', np.mean(frame_acc_implied), '%', '±', np.std(frame_acc_implied), '%', file=cv_log)
            print('Test frame acc implied:', np.mean(frame_acc_implied), '%', '±', np.std(frame_acc_implied), '%')

            print('Test frame acc RB:', np.mean(frame_acc_RB), '%', '±', np.std(frame_acc_RB), '%', file=cv_log)
            print('Test frame acc RB:', np.mean(frame_acc_RB), '%', '±', np.std(frame_acc_RB), '%')
            print('Test frame acc RB implied:', np.mean(frame_acc_RB_implied), '%', '±', np.std(frame_acc_RB_implied), '%',
                  file=cv_log)
            print('Test frame acc RB implied:', np.mean(frame_acc_RB_implied), '%', '±', np.std(frame_acc_RB_implied), '%')
            # statistical analysis
            a_predict_FB_RB_all_flat = list(itertools.chain.from_iterable(a_predict_FB_RB_all))
            a_gt_FB_all_flat = list(itertools.chain.from_iterable(a_gt_FB_all))
            a_correct_mark_all_RB_flat = list(itertools.chain.from_iterable(a_correct_mark_all_RB))
            a_correct_mark_all_ML_flat = list(itertools.chain.from_iterable(a_correct_mark_all_ML))
            output_accuracy_for_each_reason(a_counter, a_counter_correct_RB_implied, a_all_RB_reasons,
                                            a_correct_mark_all_RB_flat, NCT_bass_acc, NCT_bass_percentage, NCT_upper_acc,
                                            NCT_upper_percentage, FB_already_labeled_acc, FB_already_labeled_percentage,
                                            sixteenth_note_acc, sixteenth_note_percentage, empty_FB_acc,
                                            empty_FB_percentage, NCT_bass_err, NCT_upper_err, FB_already_labeled_err,
                                            sixteenth_note_err, empty_FB_err, cv_log, [], [],
                                            [])
            output_accuracy_for_each_reason(a_counter, a_counter_correct_implied, a_all_ML_reasons,
                                            a_correct_mark_all_ML_flat, NCT_bass_acc_ML, NCT_bass_percentage_ML, NCT_upper_acc_ML,
                                            NCT_upper_percentage_ML, FB_already_labeled_acc_ML, FB_already_labeled_percentage_ML,
                                            sixteenth_note_acc_ML, sixteenth_note_percentage_ML, empty_FB_acc_ML,
                                            empty_FB_percentage_ML, NCT_bass_err_ML, NCT_upper_err_ML, FB_already_labeled_err_ML,
                                            sixteenth_note_err_ML, empty_FB_err_ML, cv_log, ML_rule_acc, ML_rule_percentage,
                                            ML_rule_err)

            # first we need to count for which RB reason, how many got right and how many got wrong
            # NCT_bass_count = 0
            # NCT_bass_count_right = 0
            # NCT_upper_count = 0
            # NCT_upper_count_right = 0
            # FB_labelled_count = 0
            # FB_labelled_count_right = 0
            # sixteenth_count = 0
            # sixteenth_count_right = 0
            # error_counts = a_counter - a_counter_correct_RB_implied
            # empty_count = 0
            # empty_count_right = 0
            # for i, each_reason in enumerate(a_all_RB_reasons):
            #     NCT_bass_count, NCT_bass_count_right = count_each_reason_and_right_number(each_reason, NCT_bass_count,
            #                                                                               NCT_bass_count_right,
            #                                                                               a_correct_mark_all_RB_flat,
            #                                                                               'NCT bass', i)
            #     NCT_upper_count, NCT_upper_count_right = count_each_reason_and_right_number(each_reason, NCT_upper_count,
            #                                                                               NCT_upper_count_right,
            #                                                                               a_correct_mark_all_RB_flat,
            #                                                                               'NCT upper voices', i)
            #     FB_labelled_count, FB_labelled_count_right = count_each_reason_and_right_number(each_reason,
            #                                                                                     FB_labelled_count,
            #                                                                                     FB_labelled_count_right,
            #                                                                                     a_correct_mark_all_RB_flat,
            #                                                                                     'FB already labeled', i)
            #     sixteenth_count, sixteenth_count_right = count_each_reason_and_right_number(each_reason, sixteenth_count, sixteenth_count_right, a_correct_mark_all_RB_flat, '16th (or shorter) note slice ignored', i)
            #     empty_count, empty_count_right = count_each_reason_and_right_number(each_reason, empty_count, empty_count_right, a_correct_mark_all_RB_flat, [''] * len(each_reason), i)
            # NCT_bass_acc.append((NCT_bass_count_right/NCT_bass_count) * 100)
            # NCT_bass_percentage.append(NCT_bass_count/a_counter * 100)
            # NCT_upper_acc.append((NCT_upper_count_right / NCT_upper_count) * 100)
            # NCT_upper_percentage.append(NCT_upper_count/a_counter * 100)
            # FB_already_labeled_acc.append((FB_labelled_count_right/FB_labelled_count) * 100)
            # FB_already_labeled_percentage.append(FB_labelled_count/a_counter * 100)
            # sixteenth_note_acc.append((sixteenth_count_right/sixteenth_count) * 100)
            # sixteenth_note_percentage.append(sixteenth_count/a_counter * 100)
            # empty_FB_acc.append((empty_count_right / empty_count) * 100)
            # empty_FB_percentage.append(empty_count/a_counter * 100)
            # NCT_bass_err.append((NCT_bass_count - NCT_bass_count_right)/error_counts * 100)
            # NCT_upper_err.append((NCT_upper_count - NCT_upper_count_right) / error_counts * 100)
            # FB_already_labeled_err.append((FB_labelled_count - FB_labelled_count_right) / error_counts * 100)
            # sixteenth_note_err.append((sixteenth_count - sixteenth_count_right) / error_counts * 100)
            # empty_FB_err.append((empty_count - empty_count_right) / error_counts * 100)
            # print('NCT bass accuracy:', np.mean(NCT_bass_acc), '%', '±', np.std(NCT_bass_acc), '%;', 'percentage is:', np.mean(NCT_bass_percentage), '%', '±', np.std(NCT_bass_percentage), '%;')
            # print('NCT bass accuracy:', np.mean(NCT_bass_acc), '%', '±', np.std(NCT_bass_acc), '%', 'percentage is:', np.mean(NCT_bass_percentage), '%', '±', np.std(NCT_bass_percentage), '%;', file=cv_log)
            # print('NCT upper accuracy:', np.mean(NCT_upper_acc), '%', '±', np.std(NCT_upper_acc), '%', 'percentage is:', np.mean(NCT_upper_percentage), '%', '±', np.std(NCT_upper_percentage), '%;')
            # print('NCT upper cccuracy:', np.mean(NCT_upper_acc), '%', '±', np.std(NCT_upper_acc), '%', 'percentage is:', np.mean(NCT_upper_percentage), '%', '±', np.std(NCT_upper_percentage), '%;', file=cv_log)
            # print('FB already labeled accuracy:', np.mean(FB_already_labeled_acc), '%', '±', np.std(FB_already_labeled_acc), '%', 'percentage is:', np.mean(FB_already_labeled_percentage), '%', '±', np.std(FB_already_labeled_percentage), '%;')
            # print('FB already labeled accuracy:', np.mean(FB_already_labeled_acc), '%', '±', np.std(FB_already_labeled_acc), 'percentage is:', np.mean(FB_already_labeled_percentage), '%', '±', np.std(FB_already_labeled_percentage), '%;', file=cv_log)
            # print('16th note no FB accuracy:', np.mean(sixteenth_note_acc), '%', '±', np.std(sixteenth_note_acc), 'percentage is:', np.mean(sixteenth_note_percentage), '%', '±', np.std(sixteenth_note_percentage), '%;')
            # print('16th note no FB accuracy:', np.mean(sixteenth_note_acc), '%', '±', np.std(sixteenth_note_acc), 'percentage is:', np.mean(sixteenth_note_percentage), '%', '±', np.std(sixteenth_note_percentage), '%;', file=cv_log)
            # print('empty FB Rule accuracy:', np.mean(empty_FB_acc), '%', '±', np.std(empty_FB_acc), 'percentage is:', np.mean(empty_FB_percentage), '%', '±', np.std(empty_FB_percentage), '%;')
            # print('empty FB Rule accuracy:', np.mean(empty_FB_acc), '%', '±', np.std(empty_FB_acc), 'percentage is:', np.mean(empty_FB_percentage), '%', '±', np.std(empty_FB_percentage), '%;', file=cv_log)
            # print('Here is the breakdown of different types of errors, among all errors:')
            # print('Here is the breakdown of different types of errors, among all errors:', file=cv_log)
            # print('% of NCT bass being wrong:', np.mean(NCT_bass_err), '%', '±', np.std(NCT_bass_err))
            # print('% of NCT bass being wrong:', np.mean(NCT_bass_err), '%', '±', np.std(NCT_bass_err), file=cv_log)
            # print('% of NCT upper being wrong:', np.mean(NCT_upper_err), '%', '±', np.std(NCT_upper_err))
            # print('% of NCT upper being wrong:', np.mean(NCT_upper_err), '%', '±', np.std(NCT_upper_err), file=cv_log)
            # print('% of FB already labeled being wrong:', np.mean(FB_already_labeled_err), '%', '±', np.std(FB_already_labeled_err), file=cv_log)
            # print('% of FB already labeled being wrong:', np.mean(FB_already_labeled_err), '%', '±', np.std(FB_already_labeled_err))
            # print('% of 16th note no FB being wrong:', np.mean(sixteenth_note_err), '%', '±', np.std(sixteenth_note_err))
            # print('% of 16th note no FB being wrong:', np.mean(sixteenth_note_err), '%', '±', np.std(sixteenth_note_err), file=cv_log)
            # print('% of empty FB Rule being wrong:', np.mean(empty_FB_err), '%', '±', np.std(empty_FB_err))
            # print('% of empty FB Rule being wrong:', np.mean(empty_FB_err), '%', '±', np.std(empty_FB_err), file = cv_log)
            # # for i, each_answer in a_correct_mark_all_RB_flat:
            # #     if a_correct_mark_all_RB_flat[i] != '✓_' and a_correct_mark_all_RB_flat[i] != '✓':
            # #         error_counts += 1
            # #         if a_gt_FB_all_flat[i] == []:

def output_accuracy_for_each_reason(a_counter, a_counter_correct_implied, a_all_reasons, a_correct_mark_all_flat, NCT_bass_acc, NCT_bass_percentage, NCT_upper_acc, NCT_upper_percentage, FB_already_labeled_acc, FB_already_labeled_percentage, sixteenth_note_acc, sixteenth_note_percentage, empty_FB_acc, empty_FB_percentage, NCT_bass_err, NCT_upper_err, FB_already_labeled_err, sixteenth_note_err, empty_FB_err, cv_log, ML_rule_acc, ML_rule_percentage, ML_rule_err):
    NCT_bass_count = 0
    NCT_bass_count_right = 0
    NCT_upper_count = 0
    NCT_upper_count_right = 0
    FB_labelled_count = 0
    FB_labelled_count_right = 0
    sixteenth_count = 0
    sixteenth_count_right = 0
    error_counts = a_counter - a_counter_correct_implied
    empty_count = 0
    empty_count_right = 0
    ML_count = 0
    ML_count_right = 0
    for i, each_reason in enumerate(a_all_reasons):
        NCT_bass_count, NCT_bass_count_right = count_each_reason_and_right_number(each_reason, NCT_bass_count,
                                                                                  NCT_bass_count_right,
                                                                                  a_correct_mark_all_flat,
                                                                                  'NCT bass', i)
        NCT_upper_count, NCT_upper_count_right = count_each_reason_and_right_number(each_reason, NCT_upper_count,
                                                                                    NCT_upper_count_right,
                                                                                    a_correct_mark_all_flat,
                                                                                    'NCT upper voices', i)
        FB_labelled_count, FB_labelled_count_right = count_each_reason_and_right_number(each_reason,
                                                                                        FB_labelled_count,
                                                                                        FB_labelled_count_right,
                                                                                        a_correct_mark_all_flat,
                                                                                        'FB already labeled', i)
        sixteenth_count, sixteenth_count_right = count_each_reason_and_right_number(each_reason, sixteenth_count,
                                                                                    sixteenth_count_right,
                                                                                    a_correct_mark_all_flat,
                                                                                    '16th (or shorter) note slice ignored',
                                                                                    i)
        empty_count, empty_count_right = count_each_reason_and_right_number(each_reason, empty_count, empty_count_right,
                                                                            a_correct_mark_all_flat,
                                                                            [''] * len(each_reason), i)
        ML_count, ML_count_right = count_each_reason_and_right_number(each_reason, ML_count, ML_count_right,
                                                                            a_correct_mark_all_flat,
                                                                            'ML rules', i)
    NCT_bass_acc.append((NCT_bass_count_right / NCT_bass_count) * 100) if NCT_bass_count != 0 else 0
    NCT_bass_percentage.append(NCT_bass_count / a_counter * 100)
    NCT_upper_acc.append((NCT_upper_count_right / NCT_upper_count) * 100) if NCT_upper_count != 0 else 0
    NCT_upper_percentage.append(NCT_upper_count / a_counter * 100)
    FB_already_labeled_acc.append((FB_labelled_count_right / FB_labelled_count) * 100) if FB_labelled_count != 0 else 0
    FB_already_labeled_percentage.append(FB_labelled_count / a_counter * 100)
    sixteenth_note_acc.append((sixteenth_count_right / sixteenth_count) * 100) if sixteenth_count != 0 else 0
    sixteenth_note_percentage.append(sixteenth_count / a_counter * 100)
    empty_FB_acc.append((empty_count_right / empty_count) * 100) if empty_count != 0 else 0
    empty_FB_percentage.append(empty_count / a_counter * 100)
    ML_rule_acc.append((ML_count_right / ML_count) * 100) if ML_count != 0 else 0
    ML_rule_percentage.append((ML_count / a_counter) * 100)
    NCT_bass_err.append((NCT_bass_count - NCT_bass_count_right) / error_counts * 100) if error_counts != 0 else 0
    NCT_upper_err.append((NCT_upper_count - NCT_upper_count_right) / error_counts * 100) if error_counts != 0 else 0
    FB_already_labeled_err.append((FB_labelled_count - FB_labelled_count_right) / error_counts * 100) if error_counts != 0 else 0
    sixteenth_note_err.append((sixteenth_count - sixteenth_count_right) / error_counts * 100) if error_counts != 0 else 0
    empty_FB_err.append((empty_count - empty_count_right) / error_counts * 100) if error_counts != 0 else 0
    ML_rule_err.append((ML_count - ML_count_right) / error_counts * 100) if error_counts != 0 else 0
    print('NCT bass accuracy:', np.mean(NCT_bass_acc), '%', '±', np.std(NCT_bass_acc), '%;', 'percentage is:',
          np.mean(NCT_bass_percentage), '%', '±', np.std(NCT_bass_percentage), '%;')
    print('NCT bass accuracy:', np.mean(NCT_bass_acc), '%', '±', np.std(NCT_bass_acc), '%', 'percentage is:',
          np.mean(NCT_bass_percentage), '%', '±', np.std(NCT_bass_percentage), '%;', file=cv_log)
    print('NCT upper accuracy:', np.mean(NCT_upper_acc), '%', '±', np.std(NCT_upper_acc), '%', 'percentage is:',
          np.mean(NCT_upper_percentage), '%', '±', np.std(NCT_upper_percentage), '%;')
    print('NCT upper cccuracy:', np.mean(NCT_upper_acc), '%', '±', np.std(NCT_upper_acc), '%', 'percentage is:',
          np.mean(NCT_upper_percentage), '%', '±', np.std(NCT_upper_percentage), '%;', file=cv_log)
    print('FB already labeled accuracy:', np.mean(FB_already_labeled_acc), '%', '±', np.std(FB_already_labeled_acc),
          '%', 'percentage is:', np.mean(FB_already_labeled_percentage), '%', '±',
          np.std(FB_already_labeled_percentage), '%;')
    print('FB already labeled accuracy:', np.mean(FB_already_labeled_acc), '%', '±', np.std(FB_already_labeled_acc),
          'percentage is:', np.mean(FB_already_labeled_percentage), '%', '±', np.std(FB_already_labeled_percentage),
          '%;', file=cv_log)
    print('16th note no FB accuracy:', np.mean(sixteenth_note_acc), '%', '±', np.std(sixteenth_note_acc),
          'percentage is:', np.mean(sixteenth_note_percentage), '%', '±', np.std(sixteenth_note_percentage), '%;')
    print('16th note no FB accuracy:', np.mean(sixteenth_note_acc), '%', '±', np.std(sixteenth_note_acc),
          'percentage is:', np.mean(sixteenth_note_percentage), '%', '±', np.std(sixteenth_note_percentage), '%;',
          file=cv_log)
    print('empty FB Rule accuracy:', np.mean(empty_FB_acc), '%', '±', np.std(empty_FB_acc), 'percentage is:',
          np.mean(empty_FB_percentage), '%', '±', np.std(empty_FB_percentage), '%;')
    print('empty FB Rule accuracy:', np.mean(empty_FB_acc), '%', '±', np.std(empty_FB_acc), 'percentage is:',
          np.mean(empty_FB_percentage), '%', '±', np.std(empty_FB_percentage), '%;', file=cv_log)
    print('ML rule accuracy', np.mean(ML_rule_acc), '%', '±', np.std(ML_rule_acc), 'percentage is:',
          np.mean(ML_rule_percentage), '%', '±', np.std(ML_rule_percentage), '%;')
    print('ML rule accuracy', np.mean(ML_rule_acc), '%', '±', np.std(ML_rule_acc), 'percentage is:',
          np.mean(ML_rule_percentage), '%', '±', np.std(ML_rule_percentage), '%;', file=cv_log)
    print('Here is the breakdown of different types of errors, among all errors:')
    print('Here is the breakdown of different types of errors, among all errors:', file=cv_log)
    print('% of NCT bass being wrong:', np.mean(NCT_bass_err), '%', '±', np.std(NCT_bass_err))
    print('% of NCT bass being wrong:', np.mean(NCT_bass_err), '%', '±', np.std(NCT_bass_err), file=cv_log)
    print('% of NCT upper being wrong:', np.mean(NCT_upper_err), '%', '±', np.std(NCT_upper_err))
    print('% of NCT upper being wrong:', np.mean(NCT_upper_err), '%', '±', np.std(NCT_upper_err), file=cv_log)
    print('% of FB already labeled being wrong:', np.mean(FB_already_labeled_err), '%', '±',
          np.std(FB_already_labeled_err), file=cv_log)
    print('% of FB already labeled being wrong:', np.mean(FB_already_labeled_err), '%', '±',
          np.std(FB_already_labeled_err))
    print('% of 16th note no FB being wrong:', np.mean(sixteenth_note_err), '%', '±', np.std(sixteenth_note_err))
    print('% of 16th note no FB being wrong:', np.mean(sixteenth_note_err), '%', '±', np.std(sixteenth_note_err),
          file=cv_log)
    print('% of empty FB Rule being wrong:', np.mean(empty_FB_err), '%', '±', np.std(empty_FB_err))
    print('% of empty FB Rule being wrong:', np.mean(empty_FB_err), '%', '±', np.std(empty_FB_err), file=cv_log)
    print('% of ML rule being wrong:', np.mean(ML_rule_err), '%', '±', np.std(ML_rule_err))
    print('% of ML rule being wrong:', np.mean(ML_rule_err), '%', '±', np.std(ML_rule_err), file=cv_log)
    print('----------------------------------------------------------------------')


def count_each_reason_and_right_number(each_reason, count, count_right, a_correct_mark_all_RB_flat, reason_to_check, i):
    if reason_to_check in each_reason or reason_to_check == each_reason:
        count += 1
        if a_correct_mark_all_RB_flat[i] == '✓_' or a_correct_mark_all_RB_flat[i] == '✓':
            count_right += 1
    return count, count_right


def train_and_predict_non_chord_tone(layer, nodes, windowsize, portion, modelID, ts, bootstraptime, sign, augmentation,
                                     cv, pitch_class, ratio, input, output, balanced, outputtype,
                                     inputtype, predict, exclude=[]):
    print('Step 5: Training and testing the machine learning models')
    id_sum = find_id(output, '', ['099', '193', '210', '345', '053', '071', '104', '133', '182', '227',
                               '232', '238', '243', '245', '259', '261', '271', '294', '346', '239', '282', '080',
                               '121', '136', '137', '139', '141', '156', '179', '201', '247', '260', '272', '275',
                               '278', '289', '308', '333', '365'])  # get 3 digit id of the chorale
    num_of_chorale = len(id_sum)
    train_num = num_of_chorale - int(round((num_of_chorale * (1 - ratio) / 2))) * 2
    # train_num = int(num_of_chorale * ratio)
    test_num = int((num_of_chorale - train_num) / 2)
    # keys, music21 = determine_middle_name(augmentation, sign, portion)
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
    cvscores_chord_tone = []
    cvscores_test_chord_tone = []
    cvscores_percentage_of_NCT_per_slice = []
    tp = []
    tn = []
    fp = []
    fn = []
    frame_acc = []
    frame_acc_2 = [] # Use it without generating XML, and also use it to cross validate the one above
    chord_acc = []
    chord_acc_gt = []
    chord_tone_acc = [] # chord inferral ML model accuracy
    chord_tone_acc_gt = [] # chord inferral ML model accuracy using GT NCTs
    direct_harmonic_analysis_acc = []
    chord_acc_vote = []
    chord_acc_nat = []
    chord_acc_nat_spot_checking = []
    percentage_of_agreements_for_chord_inferral_algorithms = []
    batch_size = 256
    epochs = 500
    if modelID == 'DNN':
        patience = 50
    else:
        patience = 20
    # extension2 = 'batch_size' + str(batch_size) + 'epochs' + str(epochs) + 'patience' + str(
    #     patience) + 'bootstrap' + str(bootstraptime) + 'balanced' + str(balanced)
    print('Loading data...')
    if exclude == []:
        extension = sign + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21 + '_' + 'training' + str(
            train_num)
    else:
        extension = sign + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21 + '_' + 'training' + str(
            train_num) + '_39'
    timestep = ts
    HIDDEN_NODE = nodes
    MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                 str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                 + str(timestep) + extension
    MODEL_NAME_chord_tone = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                            str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                            + str(timestep) + extension + '_chord_tone'
    print('Loading data...')
    print('Build model...')
    if not os.path.isdir(os.path.join('.', 'ML_result', sign)):
        os.mkdir(os.path.join('.', 'ML_result', sign))
    if not os.path.isdir(os.path.join('.', 'ML_result', sign, MODEL_NAME)):
        os.mkdir(os.path.join('.', 'ML_result', sign, MODEL_NAME))
    cv_log = open(os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+')  + 'predict.txt', 'w')
    csv_logger = CSVLogger(os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+')  + 'predict_log.csv',
                           append=True, separator=';')
    csv_logger_chord_tone = CSVLogger(os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+')  + '_chord_tone_' + 'predict_log.csv',
                           append=True, separator=';')
    csv_logger_direct_harmonic_analysis = CSVLogger(
        os.path.join('.', 'ML_result', sign, MODEL_NAME, 'cv_log+') +  '_direct_harmonic_analysis_' + 'predict_log.csv',
        append=True, separator=';')
    error_list = []  # save all the errors to calculate frequencies
    for times in range(cv):
        # if times != 0:
        #     continue
        MODEL_NAME = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                     str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                     + str(timestep) + extension  + '_cv_' + str(times + 1)
        MODEL_NAME_chord_tone = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                     str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                     + str(timestep) + extension + '_cv_' + str(times + 1) + '_chord_tone'
        MODEL_NAME_direct_harmonic_analysis = str(layer) + 'layer' + str(nodes) + modelID + 'window_size' + \
                                str(windowsize) + '_' + str(windowsize + 1) + 'training_data' + str(portion) + 'timestep' \
                                + str(timestep) + extension + '_cv_' + str(times + 1) + '_direct_harmonic_analysis'
        FOLDER_NAME = 'MODEL'
        train_id, valid_id, test_id = get_id(id_sum, num_of_chorale, times)
        if exclude != []:
            train_id.extend(test_id)
            test_id = exclude # Swap the test id into the 39 ones
        train_num = len(train_id)
        valid_num = len(valid_id)
        test_num = len(test_id)
        if outputtype.find('_pitch_class') == -1:
            valid_yy = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                          sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        else:
            valid_yy = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                          sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'Y')
        valid_xx = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                      sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        valid_xx_chord_tone = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize + 1, ts,
                                      os.path.join('.', 'data_for_ML', sign,
                                                   sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'C')
        valid_yy_chord_tone = generate_ML_matrix(augmentation, 'valid', valid_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                          sign) + '_y_' + 'CL' + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        if not (os.path.isfile((os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5"))):
            train_xx = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                          os.path.join('.', 'data_for_ML', sign,
                                                       sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
            if outputtype.find('_pitch_class') == -1:
                train_yy = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                              os.path.join('.', 'data_for_ML', sign,
                                                           sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
            else:
                train_yy = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                              os.path.join('.', 'data_for_ML', sign,
                                                           sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                              'Y')
            train_xx_chord_tone = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize + 1, ts,
                                                     os.path.join('.', 'data_for_ML', sign,
                                                                  sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                                     'C')
            train_yy_chord_tone = generate_ML_matrix(augmentation, 'train', train_id, modelID, windowsize, ts,
                                                     os.path.join('.', 'data_for_ML', sign,
                                                                  sign) + '_y_' + 'CL' + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)

            train_xx, train_yy = bootstrap_data(train_xx, train_yy, bootstraptime)
            train_xx = train_xx[
                       :int(portion * train_xx.shape[0])]  # expose the option of training only on a subset of data
            train_yy = train_yy[:int(portion * train_yy.shape[0])]
            if balanced:  # re-balance the data
                if outputtype.find("NCT") != -1:
                    # http://imbalanced-learn.org/en/stable/introduction.html#problem-statement-regarding-imbalanced-data-sets
                    train_yy_encoded = binary_decode(train_yy)
                    ros = RandomOverSampler(ratio='minority')
                    train_xx_imbalanced = train_xx
                    train_yy_imbalanced = train_yy
                    ros_statistics = ros.fit(train_xx, train_yy_encoded)
                    train_xx, train_yy_balanced = ros.fit_sample(train_xx, train_yy_encoded)

                    train_xx = train_xx[:int(1.5 * train_xx_imbalanced.shape[0])]
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
            model = train_ML_model(modelID, HIDDEN_NODE, layer, timestep, outputtype, patience, sign,
                           FOLDER_NAME, MODEL_NAME, batch_size, epochs, csv_logger, train_xx, train_yy, valid_xx,
                           valid_yy) # train the machine learning model
            if outputtype.find("NCT") != -1: # if NCT, add the training of the chord inferral model
                model_chord_tone = train_ML_model(modelID, HIDDEN_NODE, layer, timestep, outputtype, patience, sign,
                               FOLDER_NAME, MODEL_NAME_chord_tone, batch_size, epochs, csv_logger_chord_tone, train_xx_chord_tone, train_yy_chord_tone, valid_xx_chord_tone,
                               valid_yy_chord_tone)  # train the machine learning model
                model_direct_harmonic_analysis = train_ML_model(modelID, HIDDEN_NODE, layer, timestep, 'CL', patience, sign,
                               FOLDER_NAME, MODEL_NAME_direct_harmonic_analysis, batch_size, epochs, csv_logger_direct_harmonic_analysis,
                               train_xx, train_yy_chord_tone, valid_xx,
                               valid_yy_chord_tone)  # train the direct harmonic analysis model
        # visualize the result and put into file
        test_xx = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                      sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        test_xx_no_window = generate_ML_matrix(augmentation, 'test', test_id, modelID, 0, ts,
                                     os.path.join('.', 'data_for_ML', sign,
                                                  sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        test_xx_only_pitch = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                    sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'Y')
        test_xx_only_pitch_no_window = generate_ML_matrix(augmentation, 'test', test_id, modelID, 0, 0,
                                                os.path.join('.', 'data_for_ML', sign,
                                                             sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                                'Y') # we need to first reconstruct the feature
        # vector and then turns into 3D
        if outputtype.find('_pitch_class') == -1:
            test_yy = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                        sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        else:
            test_yy = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                    sign) + '_y_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21, 'Y')
        test_yy_chord_label = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts, os.path.join('.', 'data_for_ML', sign,
                                                                                    sign) + '_y_' + 'CL' + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        test_xx_chord_tone = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize + 1, ts,
                                                 os.path.join('.', 'data_for_ML', sign,
                                                              sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                                 'C')
        test_xx_chord_tone_no_window = generate_ML_matrix(augmentation, 'test', test_id, modelID, 0, ts,
                                                os.path.join('.', 'data_for_ML', sign,
                                                             sign) + '_x_' + outputtype + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21,
                                                'C')
        test_yy_chord_tone = generate_ML_matrix(augmentation, 'test', test_id, modelID, windowsize, ts,
                                                 os.path.join('.', 'data_for_ML', sign,
                                                              sign) + '_y_' + 'CL' + pitch_class + inputtype + '_New_annotation_' + keys + '_' + music21)
        if outputtype.find("CL") != -1:
            if modelID != "SVM":
                model = load_model(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5")
                predict_y = model.predict_classes(test_xx, verbose=0)
            elif modelID == "SVM":
                predict_y = model.predict(test_xx)
                test_yy_int = np.asarray(onehot_decode(test_yy_chord_label))
                test_acc = accuracy_score(test_yy_int, predict_y)
        elif outputtype.find("NCT") != -1:
            if modelID != "SVM":
                model = load_model(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5")
                model_chord_tone = load_model(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME_chord_tone) + ".hdf5")
                model_direct_harmonic_analysis = load_model(
                    os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME_direct_harmonic_analysis) + ".hdf5")
                predict_y_direct_harmonic_analysis = model_direct_harmonic_analysis.predict_classes(test_xx)
            # TODO: 3.5
            else:
                predict_y_direct_harmonic_analysis = model_direct_harmonic_analysis.predict(test_xx)
            predict_y = model.predict(test_xx)  # Predict the probability for each bit of NCT

            test_yy_int = np.asarray(onehot_decode(test_yy_chord_label))
            test_acc = accuracy_score(test_yy_int, predict_y_direct_harmonic_analysis)
            predict_xx_chord_tone = list(test_xx_only_pitch_no_window)
            # for i, item in enumerate(test_xx_only_pitch_no_window):
            #      for j, item2 in enumerate(item):
            #          if int(item2) == 1:
            #              predict_xx_chord_tone[i][j] = 1 - predict_y[i][j]
            #      predict_xx_chord_tone[i] = np.concatenate((predict_xx_chord_tone[i], test_xx_chord_tone_no_window[i][12:])) # add beat feature
            # predict_xx_chord_tone_window = adding_window_one_hot(np.asarray(predict_xx_chord_tone), windowsize)
            # predict_y_chord_tone = model_chord_tone.predict_classes(predict_xx_chord_tone_window, verbose=0)
            if outputtype.find('pitch_class') == -1:
                input('Chord inferring model now is only working with 12 pitch class as output, not the 4 voices version!!!')
            for i in predict_y:  # regulate the prediction
                for j, item in enumerate(i):
                    if (item > 0.5):
                        i[j] = 1
                    else:
                        i[j] = 0


            for i, item in enumerate(test_xx_only_pitch_no_window):
                 if inputtype.find('NewOnset') != -1:
                     if modelID != 'SVM' and modelID != 'DNN' and modelID != 'DT':
                        if 'with_bass' not in pitch_class:
                            NewOnset = list(test_xx_no_window[i][-1][12:24])
                        else:
                            NewOnset = list(test_xx_no_window[i][-1][24:36])
                     else:
                         if 'with_bass' not in pitch_class:
                            NewOnset = list(test_xx_no_window[i][12:24])  # we need the onset sign of the vector
                         else:
                             NewOnset = list(test_xx_no_window[i][24:36])
                 for j, item2 in enumerate(item):
                     if int(predict_y[i][j]) == 1: # predict_y predicts NCT label for each slice
                        if int(item2) == 1: # if the there is a current pitch class and it is predicted as a NCT
                            predict_xx_chord_tone[i][j] = 0 # remove this pitch class since predict_xx should be all predicted CT
                            if inputtype.find('NewOnset') != -1:
                                NewOnset[j] = 0
                        # else:
                        #     input('there is a NCT for a non-existing pitch class?!')
                 if inputtype.find('NewOnset') != -1:
                     predict_xx_chord_tone[i] = np.concatenate(
                         (predict_xx_chord_tone[i], NewOnset))
                 if inputtype.find('3meter') != -1:
                     if modelID != 'SVM' and modelID != 'DNN' and modelID != 'DT':
                         predict_xx_chord_tone[i] = np.concatenate(
                             (predict_xx_chord_tone[i], test_xx_chord_tone_no_window[i][-1][-3:]))
                     else:
                         predict_xx_chord_tone[i] = np.concatenate((predict_xx_chord_tone[i], test_xx_chord_tone_no_window[i][-3:])) # add beat feature
                 # TODO: 3 might not be modular enough
            if modelID.find('SVM') != -1 or modelID.find('DNN') != -1 or modelID.find('DT') != -1:
                predict_xx_chord_tone_window = adding_window_one_hot(np.asarray(predict_xx_chord_tone), windowsize + 1)

            else:
                predict_xx_chord_tone_window = create_3D_data(np.asarray(predict_xx_chord_tone), ts)
            #predict_xx_chord_tone_window = adding_window_one_hot(np.asarray(predict_xx_chord_tone), windowsize + 1)
            if modelID == 'SVM' or modelID == 'DT':
                predict_y_chord_tone = model_chord_tone.predict(predict_xx_chord_tone_window) # TODO: we need to make this part modular so it can deal with all possible specs
                gt_y_chord_tone = model_chord_tone.predict(test_xx_chord_tone)
            else:
                predict_y_chord_tone = model_chord_tone.predict_classes(predict_xx_chord_tone_window,
                                                                        verbose=0)  # TODO: we need to make this part modular so it can deal with all possible specs
                gt_y_chord_tone = model_chord_tone.predict_classes(test_xx_chord_tone, verbose=0)
            correct_num2 = 0
            for i, item in enumerate(predict_y):
                if np.array_equal(item,test_yy[i]): # https://stackoverflow.com/questions/10580676/comparing-two-numpy-arrays-for-equality-element-wise
                    correct_num2 += 1
            frame_acc_2.append(((correct_num2 / predict_y.shape[0]) * 100))

        if modelID != 'SVM':
            test_yy_int = np.asarray(onehot_decode(test_yy_chord_label))
            scores = model.evaluate(valid_xx, valid_yy, verbose=0)
            scores_test = model.evaluate(test_xx, test_yy, verbose=0)
            print(' valid_acc: ', scores[1])
            cvscores.append(scores[1] * 100)
            cvscores_test.append(scores_test[1] * 100)
            if outputtype.find('NCT') != -1:
                scores_chord_tone = model_chord_tone.evaluate(valid_xx_chord_tone, valid_yy_chord_tone, verbose=0)
                scores_test_chord_tone = model_chord_tone.evaluate(test_xx_chord_tone, test_yy_chord_label, verbose=0)
                cvscores_chord_tone.append(scores_chord_tone[1] * 100)
                cvscores_test_chord_tone.append(scores_test_chord_tone[1] * 100)
        elif modelID == "SVM" or modelID == 'DT':
            cvscores.append(test_acc * 100)
            cvscores_test.append(test_acc * 100)
        # SaveModelLog.Save(MODEL_NAME, hist, model, valid_xx, valid_yy)
        with open('chord_name.txt') as f:
            chord_name = f.read().splitlines()
        with open('chord_name.txt') as f:
            chord_name2 = f.read().splitlines()  # delete all the chords which do not appear in the test set
        # print(matrix, file=cv_log)
        if outputtype.find('CL') != -1:
            for i, item in enumerate(chord_name):
                if i not in test_yy_int and i not in predict_y: # predict_y is different in NCT and CL!
                    chord_name2.remove(item)
        else:
            for i, item in enumerate(chord_name):
                if i not in test_yy_int and i not in predict_y_direct_harmonic_analysis: # predict_y is different in NCT and CL!
                    chord_name2.remove(item)
        if outputtype.find('CL') != -1:
            print(classification_report(test_yy_int, predict_y, target_names=chord_name2), file=cv_log)



        if outputtype.find("NCT") != -1:
            print(classification_report(test_yy_int, predict_y_direct_harmonic_analysis, target_names=chord_name2), file=cv_log)
            precision, recall, f1score, accuracy, true_positive, false_positive, false_negative, true_negative = evaluate_f1score(
                model, valid_xx, valid_yy, modelID)
            precision_test, recall_test, f1score_test, accuracy_test, asd, sdf, dfg, fgh = evaluate_f1score(model,
                                                                                                            test_xx,
                                                                                                            test_yy,
                                                                                                          modelID)
            num_of_NCT_slices = 0
            for i, item in enumerate(test_yy):
                if 1 in item:
                    num_of_NCT_slices += 1
            cvscores_percentage_of_NCT_per_slice.append((num_of_NCT_slices/test_yy.shape[0]) * 100)
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
        if predict == 'Y':
            # prediction put into files
            fileName, numSalamiSlices = get_predict_file_name(input, test_id, augmentation)
            sum = 0
            for i in range(len(numSalamiSlices)):
                sum += numSalamiSlices[i]
            # input(sum)
            # input(predict_y.shape[0])

            length = len(fileName)
            a_counter = 0
            a_counter_correct = 0
            a_counter_correct_chord = 0 # correct chord labels predicted by NCT approach
            a_counter_correct_chord_gt = 0 # correct chord labels predicted by the ground truth NCTs
            a_counter_correct_chord_tone = 0 # correct chord labels predicted by the chord inferral ML model
            a_counter_correct_chord_tone_gt = 0 # correct chord labels predicted by the chord inferral ML model using GT NCTs
            a_counter_correct_direct_harmonic_analysis = 0  # correct chord labels predicted by direct harmonic analysis
            a_counter_correct_chord_vote = 0 # record the correct chord labels by voting
            a_counter_correct_chord_nat = 0
            a_counter_correct_chord_nat_spot_checking = 0
            a_counter_number_of_agreements = 0 # the accumulative number of agreements over all chorales
            if not os.path.isdir(os.path.join('.', 'predicted_result', sign)):
                os.mkdir(os.path.join('.', 'predicted_result', sign))
            if not os.path.isdir(os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(windowsize + 1))):
                os.mkdir(os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID  + str(windowsize) + '_' + str(windowsize + 1)))

            f_all = open(os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID  + str(windowsize) + '_' + str(windowsize + 1), 'ALTOGETHER') + '.txt', 'w')  # create this file to track every type of mistakes

            for i in range(length):
                print(fileName[i][:-4], file=f_all)
                print(fileName[i][-7:-4])
                if fileName[i][-7:-4] == '099':
                    print('debug')

                num_salami_slice = numSalamiSlices[i]
                if exclude != []:
                    for fileName_Sam in os.listdir(os.path.join(output, 'Nat_GT')):
                        if fileName[i][-7:-4] in fileName_Sam: # found the Nat's annotation, load it
                            f_nat = open(os.path.join(output, 'Nat_GT', fileName_Sam), 'r')
                            chord_label_list_Nat = []
                            for chord_nat in f_nat.readlines():
                                if augmentation == 'Y':  # transpose these labels back to the original key
                                    transposed_interval, root = find_tranposed_interval(fileName[i])
                                    transposed_result = transpose_chord(transposed_interval, chord_nat.strip())
                                    chord_label_list_Nat.append(transposed_result)
                                else:  # don't transpose, so the annotation is either in C major or A minor
                                    chord_label_list_Nat.append(chord_nat.strip())
                            break
                correct_num = 0 # Record either the correct slice/chord in direct harmonic analysis or NCT approach
                correct_num_chord = 0 # record the correct predicted chord labels from NCT approach
                correct_num_chord_gt = 0 # record the correct predicted chord labels from the ground truth NCTs
                correct_num_chord_tone = 0 # record the correct predicted chord labels from the chord inferral ML model
                correct_num_chord_tone_gt = 0 # record the correct predicted chord labels from the chord inferral ML model using GT NCTs
                correct_num_direct_harmonic_analysis = 0  # record the correct predicted chord labels from direct harmonic analysis
                correct_num_vote = 0
                correct_num_nat = 0
                correct_num_nat_spot_checking = 0
                #num_of_disagreement = [] # record the number of disagreement across all chord inferring algorithms
                num_of_agreement_per_chorale = 0
                s = converter.parse(os.path.join(input, fileName[i]))  # the source musicXML file
                # s = remove_concert_pitch_voices(s)
                s_bb = converter.parse(os.path.join(input, fileName[i]))
                # s_bb = remove_concert_pitch_voices(s_bb)
                s_exclamation = converter.parse(os.path.join(input, fileName[i]))
                # s_exclamation = remove_concert_pitch_voices(s_exclamation)
                sChords = s.chordify()
                sChords_bb = s_bb.chordify()
                sChords_exclamation = s_exclamation.chordify()
                s.insert(0, sChords)
                s_bb.insert(0, sChords_bb)
                s_exclamation.insert(0, sChords_exclamation)
                chord_tone_list = []  # store all the chord tones predicted by the model
                chord_tone_list_gt = []  # store all the chord tones by the ground truth
                chord_label_list = []  # store all the chord labels predicted by the model
                chord_label_list_gt = [] # store the ground truth chord label
                chord_label_list_gt_infer = [] # store the inferred chord label by ground truth NCTs

                all_answers_per_chorale = [{} for j in range(1000)] # create an empty 2-d list that has 1000 slots to save results
                for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    thisChord_bb = sChords_bb.recurse().getElementsByClass('Chord')[j]
                    thisChord_exclamation = sChords_exclamation.recurse().getElementsByClass('Chord')[j]
                    #num_of_disagreement.append(0) # we don't have disagreement at this point
                    thisChord.closedPosition(forceOctave=4, inPlace=True)
                    thisChord_bb.closedPosition(forceOctave=4, inPlace=True)
                    thisChord_exclamation.closedPosition(forceOctave=4, inPlace=True)
                    if exclude == []: # we need the voting of Nat's annotation when it is "GT"
                        all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_name[test_yy_int[a_counter]]))] = all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_name[test_yy_int[a_counter]])), 0) + 1 # add GT
                    else: # When Sam annotation is the GT, add Nat's annotation in a different way
                        # Adding Nat's annotation the first means whenever there is a tie, choose Nat's tie!
                        all_answers_per_chorale[j][
                            unify_GTChord_and_inferred_chord(
                                translate_chord_name_into_music21(chord_label_list_Nat[j]))] = \
                            all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(
                                translate_chord_name_into_music21(chord_label_list_Nat[j])), 0) + 1  # Add Nat's
                    all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_name[predict_y_chord_tone[a_counter]]))] = all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_name[predict_y_chord_tone[a_counter]])), 0) + 1
                          # add direct harmonic analysis
                    # all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                    #         chord_name[gt_y_chord_tone[a_counter]]))] = all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                    #         chord_name[gt_y_chord_tone[a_counter]])), 0) + 1
                    #       # add chord inferral (ML) model
                    all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                            chord_name[predict_y_direct_harmonic_analysis[a_counter]]))] = all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                            chord_name[predict_y_direct_harmonic_analysis[a_counter]])), 0) + 1
                          # add chord inferral (ML) model from gt NCTs

                    if outputtype == 'CL':
                        if j == 0:
                            thisChord.addLyric('Grouth truth chord label: ' + chord_name[test_yy_int[a_counter]])
                            thisChord_bb.addLyric(chord_name[test_yy_int[a_counter]])
                            #thisChord.addLyric('Direct harmonic analysis chord label: ' +chord_name[predict_y[a_counter]])
                        else:
                            thisChord.addLyric(chord_name[test_yy_int[a_counter]])
                            thisChord_bb.addLyric(chord_name[test_yy_int[a_counter]])
                            #thisChord.addLyric(chord_name[predict_y[a_counter]])
                        if test_yy_int[a_counter] == predict_y[a_counter] or harmony.ChordSymbol(translate_chord_name_into_music21(chord_name[predict_y[a_counter]])).orderedPitchClasses == harmony.ChordSymbol(translate_chord_name_into_music21(chord_name[test_yy_int[a_counter]])).orderedPitchClasses:
                            correct_num += 1
                            print(chord_name[predict_y[a_counter]], end=' ', file=f_all)
                            if j == 0:
                                thisChord.addLyric(
                                    'Direct harmonic analysis chord label: ' + chord_name[predict_y[a_counter]] + '✓')
                            else:
                                thisChord.addLyric(chord_name[predict_y[a_counter]] + '✓')
                            #thisChord.addLyric('✓')
                        else:
                            print(chord_name[test_yy_int[a_counter]] + '->' + chord_name[predict_y[a_counter]], end=' ',
                                  file=f_all)
                            error_list.append(chord_name[test_yy_int[a_counter]] + '->' + chord_name[predict_y[a_counter]])
                            if j == 0:
                                thisChord.addLyric(
                                    'Direct harmonic analysis chord label: ' + chord_name[predict_y[a_counter]])
                            else:
                                thisChord.addLyric(chord_name[predict_y[a_counter]])
                    elif outputtype.find("NCT") != -1:
                        if j == 0:
                            thisChord.addLyric('Grouth truth chord label: ' + unify_GTChord_and_inferred_chord(chord_name[test_yy_int[a_counter]]))  # the first line is the original GT
                            thisChord_bb.addLyric(chord_name[test_yy_int[a_counter]])
                        # This insert a lane of chord inferral results
                            #thisChord.addLyric('Chord inferral (ML) chord label: ' + chord_name[predict_y_chord_tone[a_counter]])
                        else:
                            thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_name[test_yy_int[a_counter]]))  # the first line is the original GT
                            thisChord_bb.addLyric(unify_GTChord_and_inferred_chord(chord_name[test_yy_int[a_counter]]))
                            # This insert a lane of chord inferral results
                            #thisChord.addLyric(chord_name[predict_y_chord_tone[a_counter]])
                        if test_yy_int[a_counter] == predict_y_chord_tone[a_counter] or harmony.ChordSymbol(
                                translate_chord_name_into_music21(
                                    chord_name[
                                        predict_y_chord_tone[a_counter]])).orderedPitchClasses == harmony.ChordSymbol(
                            translate_chord_name_into_music21(
                                chord_name[test_yy_int[a_counter]])).orderedPitchClasses:
                            correct_num_chord_tone += 1
                            print(chord_name[predict_y_chord_tone[a_counter]], end=' ', file=f_all)
                            if j == 0:
                                thisChord.addLyric(
                                    'Chord inferral (ML) chord label: ' + unify_GTChord_and_inferred_chord(chord_name[predict_y_chord_tone[a_counter]]) + '✓')
                            else:
                                thisChord.addLyric(
                                    unify_GTChord_and_inferred_chord(chord_name[
                                        predict_y_chord_tone[a_counter]]) + '✓')
                            #thisChord.addLyric('✓')
                        else:
                            #num_of_disagreement[j] += 1
                            if j == 0:
                                thisChord.addLyric('Chord inferral (ML) chord label: ' +
                                                   unify_GTChord_and_inferred_chord(chord_name[predict_y_chord_tone[a_counter]]))
                            else:
                                thisChord.addLyric(
                                    unify_GTChord_and_inferred_chord(chord_name[
                                        predict_y_chord_tone[a_counter]]))

                        # This insert a lane of ML chord inferral results from GT NCT labels
                        # if test_yy_int[a_counter] == gt_y_chord_tone[a_counter] or harmony.ChordSymbol(
                        #         translate_chord_name_into_music21(
                        #             chord_name[
                        #                 gt_y_chord_tone[a_counter]])).orderedPitchClasses == harmony.ChordSymbol(
                        #     translate_chord_name_into_music21(
                        #         chord_name[test_yy_int[a_counter]])).orderedPitchClasses:
                        #     correct_num_chord_tone_gt += 1
                        #     print(chord_name[gt_y_chord_tone[a_counter]], end=' ', file=f_all)
                        #     if j == 0:
                        #         thisChord.addLyric(
                        #             'Chord inferral (ML) chord label from GT NCTs: ' + unify_GTChord_and_inferred_chord(chord_name[gt_y_chord_tone[a_counter]]) + '✓')
                        #     else:
                        #         thisChord.addLyric(
                        #             unify_GTChord_and_inferred_chord(chord_name[
                        #                 gt_y_chord_tone[a_counter]]) + '✓')
                        #     #thisChord.addLyric('✓')
                        # else:
                        #     num_of_disagreement[j] += 1
                        #     if j == 0:
                        #         thisChord.addLyric('Chord inferral (ML) chord label from GT NCTs: ' +
                        #                            unify_GTChord_and_inferred_chord(chord_name[gt_y_chord_tone[a_counter]]))
                        #     else:
                        #         thisChord.addLyric(
                        #             unify_GTChord_and_inferred_chord(chord_name[
                        #                 gt_y_chord_tone[a_counter]]))
                        # This insert a lane of direct harmonic analysis results

                        if test_yy_int[a_counter] == predict_y_direct_harmonic_analysis[a_counter] or harmony.ChordSymbol(
                                translate_chord_name_into_music21(
                                    chord_name[
                                        predict_y_direct_harmonic_analysis[a_counter]])).orderedPitchClasses == harmony.ChordSymbol(
                            translate_chord_name_into_music21(
                                chord_name[test_yy_int[a_counter]])).orderedPitchClasses:
                            correct_num_direct_harmonic_analysis += 1
                            print(chord_name[predict_y_direct_harmonic_analysis[a_counter]], end=' ', file=f_all)
                            if j == 0:
                                thisChord.addLyric('Direct harmonic analysis chord label: ' + unify_GTChord_and_inferred_chord(chord_name[
                                    predict_y_direct_harmonic_analysis[a_counter]]) + '✓')
                            else:
                                thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_name[predict_y_direct_harmonic_analysis[a_counter]]) + '✓')
                            #thisChord.addLyric('✓')
                        else:
                            #num_of_disagreement[j] += 1
                            if j == 0:
                                thisChord.addLyric('Direct harmonic analysis chord label: ' + unify_GTChord_and_inferred_chord(chord_name[
                                    predict_y_direct_harmonic_analysis[a_counter]]))
                            else:
                                thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_name[predict_y_direct_harmonic_analysis[a_counter]]))
                        chord_label_list_gt.append(chord_name[test_yy_int[a_counter]])
                        # pitch spelling does not affect the final results
                        gt = test_yy[a_counter]
                        prediction = predict_y[a_counter]
                        correct_bit = 0
                        for ii in range(len(gt)):
                            if (gt[ii] == prediction[ii]):  # the label is correct
                                correct_bit += 1
                        dimension = test_xx_only_pitch.shape[1]

                        realdimension = int(dimension / (2 * windowsize + 1))
                        if modelID != 'SVM' and modelID != 'DNN' and modelID != 'DT':
                            x = test_xx_only_pitch[a_counter][-1] # no window if the matrix is 3D
                        else:
                            if windowsize < 0:
                                x = test_xx_only_pitch[a_counter]
                            else:
                                x = test_xx_only_pitch[a_counter][realdimension * windowsize:realdimension * (windowsize + 1)]
                        chord_tone_gt = output_NCT_to_XML(x, gt, thisChord, outputtype)
                        chord_tone = output_NCT_to_XML(x, prediction, thisChord, outputtype)
                        if (correct_bit == len(gt)):
                            correct_num += 1
                            #thisChord.addLyric('✓')
                        # else:
                        #     thisChord.addLyric(' ')
                        chord_tone_list, chord_label_list = infer_chord_label1(thisChord, chord_tone, chord_tone_list,
                                                                               chord_label_list)
                        chord_tone_list_gt, chord_label_list_gt_infer = infer_chord_label1(thisChord, chord_tone_gt, chord_tone_list_gt,
                                                                               chord_label_list_gt_infer)
                    a_counter += 1
                a_counter_correct += correct_num

                if outputtype.find("NCT") != -1: # always compare the pitch class from the lowest ones to the highest ones, so dimished chord with different inversions should always be right answers
                    for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                        thisChord_exclamation = sChords_exclamation.recurse().getElementsByClass('Chord')[j]

                        if (chord_label_list[j] == 'un-determined' or chord_label_list[j].find('interval') != -1):  # sometimes the last
                            # chord is un-determined because there are only two tones!
                            infer_chord_label2(j, thisChord, chord_label_list, chord_tone_list)  # determine the final chord
                        infer_chord_label3(j, thisChord, chord_label_list, chord_tone_list) # TODO: Look into this later: chorale 011 M6, also the function will stumble on chorale 187
                        #thisChord.addLyric(chord_label_list[j])
                        if harmony.ChordSymbol(translate_chord_name_into_music21(chord_label_list_gt[j])).orderedPitchClasses == harmony.ChordSymbol(chord_label_list[j]).orderedPitchClasses:
                            correct_num_chord += 1
                            if j == 0:
                                thisChord.addLyric('Chord inferral (RB) chord label: ' + unify_GTChord_and_inferred_chord(chord_label_list[j]) + '✓')
                            else:
                                thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_label_list[j]) + '✓')
                            #thisChord.addLyric('✓')
                        else:
                            #num_of_disagreement[j] += 1
                            if j == 0:
                                thisChord.addLyric('Chord inferral (RB) chord label: ' + unify_GTChord_and_inferred_chord(chord_label_list[j]))
                            else:
                                thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_label_list[j]))
                        if (chord_label_list_gt_infer[j] == 'un-determined' or chord_label_list_gt_infer[j].find('interval') != -1):  # sometimes the last
                            # chord is un-determined because there are only two tones!
                            infer_chord_label2(j, thisChord, chord_label_list_gt_infer, chord_tone_list_gt)  # determine the final chord
                        infer_chord_label3(j, thisChord, chord_label_list_gt_infer, chord_tone_list_gt)
                        #thisChord.addLyric(chord_label_list_gt_infer[j])
                        #print('slice number:', j, 'gt:', chord_label_list_gt[j], 'prediction:', chord_label_list[j])
                        # if harmony.ChordSymbol(translate_chord_name_into_music21(chord_label_list_gt[j])).orderedPitchClasses == harmony.ChordSymbol(chord_label_list_gt_infer[j]).orderedPitchClasses:
                        #     correct_num_chord_gt += 1
                        #     if j == 0:
                        #         thisChord.addLyric('Ground truth chord inferral (RB) chord label: ' + unify_GTChord_and_inferred_chord(chord_label_list_gt_infer[j]) + '✓')
                        #     else:
                        #         thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_label_list_gt_infer[j]) + '✓')
                        #     # thisChord.addLyric('✓')
                        # else:
                        #     num_of_disagreement[j] += 1
                        #     if j == 0:
                        #         thisChord.addLyric('Ground truth chord inferral (RB) chord label: ' + unify_GTChord_and_inferred_chord(chord_label_list_gt_infer[j]))
                        #     else:
                        #         thisChord.addLyric(unify_GTChord_and_inferred_chord(chord_label_list_gt_infer[j]))
                        # output the number of disagreement

                        all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list[j]))] = \
                        all_answers_per_chorale[j].get(unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list[j])), 0) + 1
                        # add Chord inferral (RB) chord label

                        # annotation as a part of the voting member
                        # all_answers_per_chorale[j][
                        #     unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list_gt_infer[j]))] = all_answers_per_chorale[
                        #                                                                            j].get(
                        #     unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list_gt_infer[j])),
                        #     0) + 1  # add Ground truth chord inferral (RB) chord label
                        sorted_result = sorted(all_answers_per_chorale[j].items(), key=lambda d: d[1], reverse=True)
                        #print(sorted_result)
                        if sorted_result[0] == sorted_result[-1]: # Only one answer, agreed
                            num_of_agreement_per_chorale += 1
                            thisChord.addLyric(' ')
                            thisChord_exclamation.addLyric(sorted_result[0][0])
                        else:
                            thisChord.addLyric('_!')
                            thisChord_exclamation.addLyric('_!')
                        #print(sorted_result)
                        if harmony.ChordSymbol(translate_chord_name_into_music21(chord_label_list_gt[j])).orderedPitchClasses == harmony.ChordSymbol(translate_chord_name_into_music21(sorted_result[0][0])).orderedPitchClasses:
                            correct_num_vote += 1
                            if j == 0:
                                thisChord.addLyric('Voting: ' + str(sorted_result[0][0]) + '✓')
                            else:
                                thisChord.addLyric(str(sorted_result[0][0]) + '✓')
                        else:
                            if j == 0:
                                thisChord.addLyric('Voting: ' + str(sorted_result[0][0]))
                            else:
                                thisChord.addLyric(str(sorted_result[0][0]))
                        if exclude != []:

                            if harmony.ChordSymbol(translate_chord_name_into_music21(
                                chord_label_list_gt[j])).orderedPitchClasses == harmony.ChordSymbol(
                                translate_chord_name_into_music21(chord_label_list_Nat[j])).orderedPitchClasses:
                                correct_num_nat += 1
                                correct_num_nat_spot_checking += 1 # Because I modify Nat's annotation for spot checking
                                if j == 0:
                                    thisChord.addLyric('Nat: ' + chord_label_list_Nat[j] + '✓')
                                    thisChord.addLyric('Nat spot checking: ' + '✓')
                                else:
                                    thisChord.addLyric(chord_label_list_Nat[j] + '✓')
                                    thisChord.addLyric('✓')
                            else:
                                if j == 0:
                                    thisChord.addLyric('Nat: ' + chord_label_list_Nat[j])
                                else:
                                    thisChord.addLyric(chord_label_list_Nat[j])
                                if sorted_result[0] != sorted_result[-1]:
                                    # this means this error can be found by spot checking
                                    correct_num_nat_spot_checking += 1
                                    if j == 0:
                                        thisChord.addLyric('Nat spot checking: ' + '✓')
                                    else:
                                        thisChord.addLyric('✓')
                                else:
                                    if j == 0:
                                        thisChord.addLyric('Nat spot checking: ' + ' ')
                                    else:
                                        thisChord.addLyric(' ')

                a_counter_correct_chord += correct_num_chord
                a_counter_correct_chord_gt += correct_num_chord_gt
                a_counter_correct_chord_tone += correct_num_chord_tone
                a_counter_correct_chord_tone_gt += correct_num_chord_tone_gt
                a_counter_correct_direct_harmonic_analysis += correct_num_direct_harmonic_analysis
                a_counter_correct_chord_vote += correct_num_vote
                a_counter_correct_chord_nat += correct_num_nat
                a_counter_correct_chord_nat_spot_checking += correct_num_nat_spot_checking
                a_counter_number_of_agreements += num_of_agreement_per_chorale
                print(end='\n', file=f_all)
                print('frame accucary: ' + str(correct_num / num_salami_slice), end='\n', file=f_all)
                print('num of correct frame answers: ' + str(correct_num) + ' number of salami slices: ' + str(num_salami_slice),
                      file=f_all)
                print('accumulative frame accucary: ' + str(a_counter_correct / a_counter), end='\n', file=f_all)
                print('chord accucary: ' + str(correct_num_chord / num_salami_slice), end='\n', file=f_all)

                print('num of correct chord answers: ' + str(correct_num_chord) + ' number of salami slices: ' + str(num_salami_slice),
                      file=f_all)
                print('chord Nat accucary: ' + str(correct_num_nat / num_salami_slice), end='\n', file=f_all)
                print('num of correct chord Nat answers: ' + str(correct_num_nat) + ' number of salami slices: ' + str(
                    num_salami_slice),
                      file=f_all)
                print('accumulative chord accucary: ' + str(a_counter_correct_chord / a_counter), end='\n', file=f_all)
                print('accumulative chord ground truth accucary: ' + str(a_counter_correct_chord_gt / a_counter), end='\n', file=f_all)
                print('accumulative chord voting accucary: ' + str(a_counter_correct_chord_vote / a_counter),
                      end='\n', file=f_all)
                print('accumulative chord Nat accucary: ' + str(a_counter_correct_chord_nat / a_counter),
                      end='\n', file=f_all)
                print('accumulative chord Nat spot checking accucary: ' + str(a_counter_correct_chord_nat_spot_checking / a_counter),
                      end='\n', file=f_all)
                print('accumulative chord inferral (ML) accucary: ' + str(a_counter_correct_chord_tone / a_counter),
                      end='\n', file=f_all)
                print('accumulative chord inferral (ML) accucary using GT NCTs: ' + str(a_counter_correct_chord_tone_gt / a_counter),
                      end='\n', file=f_all)
                print('accumulative direct harmonic analysis accucary: ' + str(a_counter_correct_direct_harmonic_analysis / a_counter),
                      end='\n', file=f_all)
                s.write('musicxml',
                        fp=os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID  + str(windowsize) + '_' + str(windowsize + 1), fileName[i][
                                                                                              :-4]) + '.xml')
                s_bb.write('musicxml',
                        fp=os.path.join('.', 'predicted_result', sign, outputtype + pitch_class + inputtype + modelID  + str(windowsize) + '_' + str(windowsize + 1),
                                        fileName[i][
                                        :-4]) + '_bb.xml')
                s_exclamation.write('musicxml',
                           fp=os.path.join('.', 'predicted_result', sign,
                                           outputtype + pitch_class + inputtype + modelID + str(windowsize) + '_' + str(windowsize + 1),
                                           fileName[i][
                                           :-4]) + '_exclamation.xml')
                # output result in musicXML
            frame_acc.append((a_counter_correct / a_counter) * 100)
            chord_acc.append((a_counter_correct_chord / a_counter) * 100)
            chord_acc_gt.append((a_counter_correct_chord_gt / a_counter) * 100)
            chord_tone_acc.append((a_counter_correct_chord_tone / a_counter) * 100)
            chord_tone_acc_gt.append((a_counter_correct_chord_tone_gt / a_counter) * 100)
            direct_harmonic_analysis_acc.append((a_counter_correct_direct_harmonic_analysis / a_counter) * 100)
            chord_acc_vote.append((a_counter_correct_chord_vote / a_counter) * 100)
            chord_acc_nat.append((a_counter_correct_chord_nat / a_counter) * 100)
            chord_acc_nat_spot_checking.append((a_counter_correct_chord_nat_spot_checking / a_counter) * 100)
            percentage_of_agreements_for_chord_inferral_algorithms.append((a_counter_number_of_agreements / a_counter) * 100)
    if predict == 'Y':
        counts = Counter(error_list)
        print(counts, file=f_all)
        f_all.close()
    print(np.mean(cvscores), np.std(cvscores))
    print(MODEL_NAME, file=cv_log)
    if modelID != 'SVM' and modelID != 'DT':
        model = load_model(os.path.join('.', 'ML_result', sign, FOLDER_NAME, MODEL_NAME) + ".hdf5")
        model.summary(print_fn=lambda x: cv_log.write(x + '\n'))  # output model struc ture into the text file
    print('valid accuracy:', np.mean(cvscores), '%', '±', np.std(cvscores), '%', file=cv_log)
    if outputtype.find("NCT") != -1:
        print('valid precision:', np.mean(pre), '%', '±', np.std(pre), '%', file=cv_log)
        print('valid recall:', np.mean(rec), '%', '±', np.std(rec), '%', file=cv_log)
        print('valid f1:', np.mean(f1), '%', '±', np.std(f1), '%', file=cv_log)
        print('valid acc (validate previous):', np.mean(acc), '%', '±', np.std(acc), '%', file=cv_log)
        print('valid tp number:', np.mean(tp), '±', np.std(tp), file=cv_log)
        print('valid fp number:', np.mean(fp), '±', np.std(fp), file=cv_log)
        print('valid fn number:', np.mean(fn), '±', np.std(fn), file=cv_log)
        print('valid tn number:', np.mean(tn), '±', np.std(tn), file=cv_log)
        if predict == 'Y':
            for i in range(len(cvscores_test)):
                print('Test f1:', i, f1_test[i], '%', 'Frame acc:', frame_acc[i], '%', 'Frame acc 2:', frame_acc_2[i], '%', 'Chord acc:', chord_acc[i], 'Chord gt acc:', chord_acc_gt[i], 'Chord tone acc:', chord_tone_acc[i], 'Chord tone gt acc:', chord_tone_acc_gt[i], 'Direct harmonic analysis acc:', direct_harmonic_analysis_acc[i], '% of agreements:', percentage_of_agreements_for_chord_inferral_algorithms[i], 'Voting acc:', chord_acc_vote[i], 'Nat acc:', chord_acc_nat[i], 'Nat acc spot checking:', chord_acc_nat_spot_checking[i], file=cv_log)
        else:
            for i in range(len(cvscores_test)):
                print('Test f1:', i, f1_test[i], '%',  'Frame acc 2:', frame_acc_2[i], '%', file=cv_log)
    elif outputtype.find("CL") != -1:
        if predict == 'Y':
            for i in range(len(cvscores_test)):
                print('Test acc:', i, cvscores_test[i], '%', 'Frame acc:', frame_acc[i], '%', file=cv_log)
        else:
            for i in range(len(cvscores_test)):
                print('Test acc:', i, cvscores_test[i], '%', file=cv_log)
    print('Test accuracy:', np.mean(cvscores_test), '%', '±', np.std(cvscores_test), '%', file=cv_log)
    if outputtype.find("CL") != -1 and predict == 'Y':
        print('Test frame acc:', np.mean(frame_acc), '%', '±', np.std(frame_acc), '%', file=cv_log)
    if outputtype.find("NCT") != -1:
        print('Test precision:', np.mean(pre_test), '%', '±', np.std(pre_test), '%', file=cv_log)
        print('Test recall:', np.mean(rec_test), '%', '±', np.std(rec_test), '%', file=cv_log)
        print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%', file=cv_log)
        print('Test f1:', np.mean(f1_test), '%', '±', np.std(f1_test), '%',)
        print('Test % of NCTs per slice:', np.mean(cvscores_percentage_of_NCT_per_slice), '%', '±', np.std(cvscores_percentage_of_NCT_per_slice), '%', file=cv_log)
        print('Test % of NCTs per slice:', np.mean(cvscores_percentage_of_NCT_per_slice), '%', '±',
              np.std(cvscores_percentage_of_NCT_per_slice), '%')
        print('Test acc:', np.mean(acc_test), '%', '±', np.std(acc_test), '%', file=cv_log)
        print('Test frame acc 2:', np.mean(frame_acc_2), '%', '±', np.std(frame_acc_2), '%', file=cv_log)
        print('Test frame acc 2:', np.mean(frame_acc_2), '%', '±', np.std(frame_acc_2), '%')
        if predict == 'Y':
            print('Test frame acc:', np.mean(frame_acc), '%', '±', np.std(frame_acc), '%', file=cv_log)
            print('Test chord acc:', np.mean(chord_acc), '%', '±', np.std(chord_acc), '%', file=cv_log)
            print('Test chord acc gt:', np.mean(chord_acc_gt), '%', '±', np.std(chord_acc_gt), '%', file=cv_log)
            print('Test chord tone acc:', np.mean(chord_tone_acc), '%', '±', np.std(chord_tone_acc), '%', file=cv_log)
            print('Test chord tone acc gt:', np.mean(chord_tone_acc_gt), '%', '±', np.std(chord_tone_acc_gt), '%', file=cv_log)
            print('Test direct harmonic analysis acc:', np.mean(direct_harmonic_analysis_acc), '%', '±', np.std(direct_harmonic_analysis_acc), '%', file=cv_log)
            print('Test % of agreements:', np.mean(percentage_of_agreements_for_chord_inferral_algorithms), '%', '±', np.std(percentage_of_agreements_for_chord_inferral_algorithms), '%', file=cv_log)
            print('Test chord acc voting:', np.mean(chord_acc_vote), '%', '±', np.std(chord_acc_vote), '%', file=cv_log)
            print('Test chord acc Nat:', np.mean(chord_acc_nat), '%', '±', np.std(chord_acc_nat), '%', file=cv_log)
            print('Test chord acc Nat spot checking:', np.mean(chord_acc_nat_spot_checking), '%', '±', np.std(chord_acc_nat_spot_checking), '%', file=cv_log)
            print('Test frame acc:', np.mean(frame_acc), '%', '±', np.std(frame_acc), '%')
            print('Test chord acc:', np.mean(chord_acc), '%', '±', np.std(chord_acc), '%')
            print('Test chord acc gt:', np.mean(chord_acc_gt), '%', '±', np.std(chord_acc_gt), '%')
            print('Test chord tone acc:', np.mean(chord_tone_acc), '%', '±', np.std(chord_tone_acc), '%')
            print('Test chord tone acc gt:', np.mean(chord_tone_acc_gt), '%', '±', np.std(chord_tone_acc_gt), '%')
            print('Test direct harmonic analysis acc:', np.mean(direct_harmonic_analysis_acc), '%', '±',
                  np.std(direct_harmonic_analysis_acc), '%')
            print('Test % of agreements:', np.mean(percentage_of_agreements_for_chord_inferral_algorithms), '%', '±', np.std(percentage_of_agreements_for_chord_inferral_algorithms), '%')
            print('Test chord acc voting:', np.mean(chord_acc_vote), '%', '±', np.std(chord_acc_vote), '%')
            print('Test chord acc Nat:', np.mean(chord_acc_nat), '%', '±', np.std(chord_acc_nat), '%')
            print('Test chord acc Nat spot checking:', np.mean(chord_acc_nat_spot_checking), '%', '±',
                  np.std(chord_acc_nat_spot_checking), '%')
    cv_log.close()

if __name__ == "__main__":
    train_and_predict_non_chord_tone(2, 200, 2, 1, 'DNN', 10)
