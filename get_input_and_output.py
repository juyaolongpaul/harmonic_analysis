from music21 import *
import os
import re
import numpy as np
dic = {}
from counter_chord_frequency import *
from adding_window_one_hot import adding_window_one_hot
from test_musicxml_gt import get_chord_tone
from random import shuffle
format = ['mid']
cwd = '.\\bach_chorales_scores\\transposed_MIDI\\'
x = []
y = []


def get_chord_line(line, sign):
    """

    :param line:
    :param replace:
    :return:
    """
    for letter in '!':
        line = line.replace(letter, '')
    #if(sign == '0'):  # now, no inversions what so ever
    line = re.sub(r'/\w[b#]*', '', line)  # remove inversions + shapr + flat
    return line


def calculate_freq(dic, line):
    """
    :param dic:
    :param line:
    :return:
    """
    for chord in line.split():
        dic.setdefault(chord, 0)
        dic[chord] += 1
    return dic


def output_freq_to_file(filename, dic):
    """

    :param filename:
    :param dic:
    :return:
    """
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    fchord = open(filename, 'w')
    total_freq = 0
    total_percentage = 0
    for word in li:
        total_freq += word[1]
    for word in li:
        print(word, end='', file=fchord)
        total_percentage += word[1] / total_freq
        print(str(word[1] / total_freq), end='', file=fchord)
        print(' total: ' + str(total_percentage), file=fchord)


def fill_in_chord_class(chord, chordclass, list):
    """

    :param chordclass: The chord class vector that needs to label
    :param list: The chord list with outputdim top freq
    :return: the modified pitch class that need to store
    """
    for i, chord2 in enumerate(list):
        if(chord == chord2):

            chordclass[i] = 1
            break
    empty = [0] * (len(list)+1)
    if(chordclass == empty):  # this is 'other' chord!
        chordclass[len(list)] = 1
    return chordclass


def y_non_chord_tone(chord, chordclass, list):
    """

    :param chordclass: The chord class vector that needs to label
    :param list: The chord list with outputdim top freq
    :return: the modified pitch class that need to store
    """
    for i, chord2 in enumerate(list):
        if(chord == chord2):

            chordclass[i] = 1
            break
    empty = [0] * (len(list)+1)
    if(chordclass == empty):  # this is 'other' chord!
        chordclass[len(list)] = 1
    return chordclass


def fill_in_pitch_class_with_bass(pitchclass, list, counter):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """

    pitchclassvoice = pitchclass
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    for i in list:
        pitchclass[i] = 1
    pitchclassvoice[0] = pitchclass
    if len(list) == 4:
        pitchclassvoice[1][list[3]] = 1  # the last voice is the bass, proved
    else:
        print('no bass?')
        counter += 1
        print(counter)
    pitchclassvoice = pitchclassvoice.ravel()
    pitchclassvoice = pitchclassvoice.tolist()
    return pitchclassvoice, counter


def fill_in_pitch_class_with_octave(list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    pitchclass = [0] * 72 # calculate by pitch_distribution
    for i in list:
        pitchclass[(i.midi-24)] = 1  # the lowest is 28, compressed
    return pitchclass


def pitch_distribution(list, counter, countermin):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        if i.midi > counter:
            counter = i.midi
            print('max', counter)
        if i.midi < countermin:
            countermin = i.midi
            print('min', countermin)
        #if(i.midi > 84):
            #input('pitch class more than 84')
    return counter, countermin


def fill_in_pitch_class_with_voice(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    pitchclassvoice = pitchclass
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))  # 2-D array

    for i, item in enumerate(list):
        pitchclassvoice[i][item] = 1
    pitchclassvoice = pitchclassvoice.ravel()
    pitchclassvoice = pitchclassvoice.tolist()
    return pitchclassvoice


def fill_in_pitch_class(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        pitchclass[i] = 1
    return pitchclass


def get_chord_list(output_dim, sign):
    dic = {}
    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if file_name.find('translated_transposed') != -1:
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line = get_chord_line(line, sign)
                #print(line)
                dic = calculate_freq(dic, line)
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    list_of_chords = []
    for i, word in enumerate(li):
        if(i == output_dim - 1):  # the last one is 'others'
            break
        list_of_chords.append(word[0])
    print (list_of_chords)  # Get the top 35 chord freq
    return list_of_chords


def add_beat_into(pitchclass, beat):
    """
    adding two dimension to the input vector, specifying whether the current slice is on/off beat.
    :return:
    """
    if(len(beat) == 1):  # on beat
        pitchclass.append(1)
        pitchclass.append(0)
    else:
        pitchclass.append(0)
        pitchclass.append(1)
    return pitchclass


def generate_data_windowing(counter1, counter2, string, string1, string2, x, y, inputdim, outputdim, windowsize, counter, countermin):
    file_counter = 0
    slice_counter = 0

    for id, fn in enumerate(os.listdir(cwd)):
        #print(fn)
        if fn[-3:] == 'mid':
            chorale_x = []
            if (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop','r')

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop','r')

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop','r')

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
            else:
                continue  # skip the file which does not have chord labels
            file_counter += 1
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            #length = len(sChords)
            for line in f.readlines():
                line = get_chord_line(line, sign)
                for chord in line.split():
                    if(chord.find('g]') != -1):
                        print(fn)
                        input('wtf is that?')
                    counter2 += 1
                    chord_class = [0] * outputdim
                    chord_class = fill_in_chord_class(chord, chord_class, list_of_chords)
                    if(counter2 == 1):
                        y = np.concatenate((y, chord_class))
                    else:
                        y = np.vstack((y, chord_class))
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

                counter1 += 1
                slice_counter += 1
                pitchClass = [0] * inputdim
                #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
                #(thisChord.pitchClasses)
                pitchClass = add_beat_into(pitchClass, thisChord.beatStr)  # add on/off beat info
                if(i == 0):
                    chorale_x = np.concatenate((chorale_x, pitchClass))
                else:
                    chorale_x = np.vstack((chorale_x, pitchClass))
            chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
            if(file_counter == 1):
                x = chorale_x_window
            else:
                x = np.concatenate((x, chorale_x_window))

    # add zero to the matrix, so that it can be divided by 50
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    yy = [0] * outputdim
    yy[-1] = 1  # there should be a chord for these artificial slices, chord is 'other'
    print('yy:', yy)
    '''while(x.shape[0] % 50 !=0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))'''
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    np.savetxt(string + string1 + string2 + '_x_windowing_' + str(windowsize) + '71-140.txt', x, fmt = '%.1e')
    np.savetxt(string + string1 + string2 + '_y_windowing_' + str(windowsize) + '71-140.txt', y, fmt = '%.1e')


def get_non_chord_tone(x,y,outputdim):
    """
    Take out chord tones, only leave with non-chord tone

    :param x:
    :param y:
    :return:
    """
    yy = [0] * len(y)
    for i in range(outputdim):
        if(x[i] == 1 and y[i] == 0):  # non-chord tone
            if(y[-1] != 1):
                yy[i] = 1
        if(y[-1] == 1):  # occasion where the chord cannot be recognized
            if(yy == [0] * len(y)):
                yy[-1] = 1
            else:
                print('x:' , x)
                print('y:' , y)
                print('yy:' , yy)
                input('error')
            break
    return yy


def get_non_chord_tone_4(x,y,outputdim, f):
    """
    Take out chord tones, only leave with non-chord tone

    :param x:
    :param y:
    :return:
    """
    pitchclass = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    nonchordpitchclassptr = [-1] * 4
    yori = y
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    for i in range(len(x) - 2):
        if(yori[-1] == 1): # broken chord, assume there is no non-chord tone!
            break
        if(x[i] == 1):  # go through the present pitch class
            yyptr += 1
            if(y[i%12] == 0):  # the present pitch class is a chord tone or not
                #print('yyptr:' , yyptr)
                if(yyptr == 4):
                    print('debug')
                yy[yyptr] = 1
                nonchordpitchclassptr[yyptr] = i%12
    if(nonchordpitchclassptr == [-1] * 4):
        print('n/a', end= ' ', file=f)
    else:
        if(2 in nonchordpitchclassptr):
            print('debug')
        for item in nonchordpitchclassptr:
            if(item != -1):
                print(pitchclass[item], end='', file=f)
        print(end=' ', file=f)
    return yy


def get_non_chord_tone_4_music21(x, y, f, thisChord):
    '''
    Getting crappy but consistent non-chord tone labels from music21 module. Four triads and five types of 7th chords
    are considered. Otherwise they are all non-chord tones.
    :param x:
    :param y:
    :return:
    '''
    allowed_qualities = [[0, 4, 7], [0, 3, 6], [0, 3, 7], [0, 4, 8], [0, 3, 6, 9], [0, 3, 6, 10], [0, 3, 7, 10],
                         [0, 4, 7, 10], [0, 4, 7, 11]]
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    nonchordpitchclassptr = [-1] * 4
    yori = y
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    if (thisChord.normalForm in allowed_qualities or thisChord.pitchedCommonName.find(
            'incomplete') != -1
            or ((thisChord.pitchedCommonName.find(
                'dominant') != -1 or thisChord.pitchedCommonName.find(
                'diminished') != -1 or thisChord.pitchedCommonName.find('major') != -1
                 or thisChord.pitchedCommonName.find(
                        'half-diminished') != -1 or thisChord.pitchedCommonName.find('minor') != -1)
                and thisChord.pitchedCommonName.find('seventh') != -1)):
        print('this slice, music21 consider no nct!')
    else:  # they are all non-chord tones
        for i in range(len(x) - 2):
            if (yori[-1] == 1):  # broken chord, assume there is no non-chord tone!
                break
            if (x[i] == 1):  # go through the present pitch class
                yyptr += 1
                #if (y[i % 12] == 0):  # the present pitch class is a chord tone or not
                    # print('yyptr:' , yyptr)
                if (yyptr == 4):
                    print('debug')
                yy[yyptr] = 1
                nonchordpitchclassptr[yyptr] = i % 12
    if (nonchordpitchclassptr == [-1] * 4):
        print('n/a', end=' ', file=f)
    else:
        if (2 in nonchordpitchclassptr):
            print('debug')
        for item in nonchordpitchclassptr:
            if (item != -1):
                print(pitchclass[item], end='', file=f)
        print(end=' ', file=f)
    return yy

def generate_data_windowing_non_chord_tone(counter1, counter2, string, string1, string2, x, y, inputdim, outputdim, windowsize, counter, countermin):
    file_counter = 0
    slice_counter = 0
    fn_total = []
    for id, fn in enumerate(os.listdir(cwd)):
        #print(fn)
        if fn[-3:] == 'mid':
            fn_total.append(fn)
    shuffle(fn_total)
    print (fn_total)
    #input('?')
    for id, fn in enumerate(fn_total):

            chorale_x = []
            if (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt','w')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt',
                             'w')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt',
                             'w')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt',
                             'w')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt',
                             'w')
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
                f_non = open('.\\useful_chord_symbols\\' + string + '\\transposed_non_chord_tone' + fn[0:3] + '.txt',
                             'w')
            else:
                continue  # skip the file which does not have chord labels
            file_counter += 1
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            #length = len(sChords)
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

                counter1 += 1
                slice_counter += 1
                pitchClass = [0] * inputdim
                #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
                #(thisChord.pitchClasses)
                pitchClass = add_beat_into(pitchClass, thisChord.beatStr)  # add on/off beat info
                if(i == 0):
                    chorale_x = np.concatenate((chorale_x, pitchClass))
                else:
                    chorale_x = np.vstack((chorale_x, pitchClass))
            chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
            if(file_counter == 1):
                x = chorale_x_window
            else:
                x = np.concatenate((x, chorale_x_window))
            slice_counter = 0  # remember what slice in order to get the pitch class info
            for line in f.readlines():
                line = get_chord_line(line, sign)
                for chord in line.split():
                    if(chord.find('g]') != -1):
                        print(fn)
                        input('wtf is that?')
                    counter2 += 1
                    #chord_class = [0] * outputdim
                    #chord_class = y_non_chord_tone(chord, chord_class, list_of_chords)
                    #chord_class = get_non_chord_tone(chorale_x[slice_counter],)
                    chord_class = get_chord_tone(chord, output_dim)
                    #chord_class = get_non_chord_tone(chorale_x[slice_counter], chord_class, output_dim)
                    chord_class = get_non_chord_tone_4(chorale_x[slice_counter], chord_class, output_dim, f_non)

                    slice_counter += 1
                    if(counter2 == 1):
                        y = np.concatenate((y, chord_class))
                    else:
                        y = np.vstack((y, chord_class))
            # save x, y into file as "ground truth"

    # add zero to the matrix, so that it can be divided by 50
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    yy = [0] * outputdim
    yy[-1] = 1  # there should be a chord for these artificial slices, chord is 'other'
    print('yy:', yy)
    '''while(x.shape[0] % 50 !=0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))'''
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    np.savetxt(string + string1 + string2 + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class.txt', x, fmt = '%.1e')
    np.savetxt(string + string1 + string2 + '_y_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class.txt', y, fmt = '%.1e')

def generate_data_windowing_no_y(counter1, string, string1, string2, x, inputdim, windowsize, counter, countermin):
    file_counter = 0
    slice_counter = 0

    for id, fn in enumerate(os.listdir(cwd)):
        #print(fn)
        if fn[-3:] == 'mid':
            chorale_x = []
            if (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                continue

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                continue
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                continue

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string1 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                continue
            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                continue

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string2 + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                continue
            else:

                file_counter += 1
                s = converter.parse(cwd + fn)
                sChords = s.chordify()
            #length = len(sChords)

                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

                    counter1 += 1
                    slice_counter += 1
                    pitchClass = [0] * inputdim
                    #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                    pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                    counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                    #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
                    #(thisChord.pitchClasses)
                    pitchClass = add_beat_into(pitchClass, thisChord.beatStr)  # add on/off beat info
                    if(i == 0):
                        chorale_x = np.concatenate((chorale_x, pitchClass))
                    else:
                        chorale_x = np.vstack((chorale_x, pitchClass))
                chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
                if(file_counter == 1):
                    x = chorale_x_window
                else:
                    x = np.concatenate((x, chorale_x_window))

    # add zero to the matrix, so that it can be divided by 50
    np.savetxt(string + string1 + string2 + '_x_windowing_' + str(windowsize) + '_230.txt', x, fmt = '%.1e')

def generate_data_windowing_non_chord_tone_new_annotation(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1, output, f2, sign):
    """
    The only difference with "generate_data_windowing_non_chord_tone"
    :param counter1:
    :param counter2:
    :param string:
    :param string1:
    :param string2:
    :param x:
    :param y:
    :param inputdim:
    :param outputdim:
    :param windowsize:
    :param counter:
    :param countermin:
    :return:
    """
    file_counter = 0
    slice_counter = 0
    fn_total = []
    for id, fn in enumerate(os.listdir(input)):
        #print(fn)
        if fn.find( 'transposed_chor') != -1 and fn[-4:] == f1:  # only look for the transposed one
            fn_total.append(fn)
    shuffle(fn_total)
    print (fn_total)
    #input('?')
    for id, fn in enumerate(fn_total):
            print(fn)
            chorale_x = []
            if (os.path.isfile(output + 'transposed_translated_' + fn[-7:-4] + sign + f2)):
                f = open(output + 'transposed_translated_' + fn[-7:-4] + sign + f2,'r')
                f_non = open(output + '\\transposed_non_chord_tone_'+ sign + fn[-7:-4] + f2,'w')
            else:
                continue  # skip the file which does not have chord labels
            file_counter += 1
            s = converter.parse(input + fn)
            sChords = s.chordify()
            slice_input = 0
            #print(slice_input)
            #length = len(sChords)
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                slice_input += 1
                counter1 += 1
                slice_counter += 1
                pitchClass = [0] * inputdim
                #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
                #(thisChord.pitchClasses)
                pitchClass = add_beat_into(pitchClass, thisChord.beatStr)  # add on/off beat info
                if(i == 0):
                    chorale_x = np.concatenate((chorale_x, pitchClass))
                else:
                    chorale_x = np.vstack((chorale_x, pitchClass))
            chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
            if(file_counter == 1):
                x = chorale_x_window
            else:
                x = np.concatenate((x, chorale_x_window))
            slice_counter = 0  # remember what slice in order to get the pitch class info
            for line in f.readlines():
                line = get_chord_line(line, sign)

                for chord in line.split():
                    if(chord.find('g]') != -1):
                        print(fn)
                        input('wtf is that?')
                    counter2 += 1
                    #chord_class = [0] * outputdim
                    #chord_class = y_non_chord_tone(chord, chord_class, list_of_chords)
                    #chord_class = get_non_chord_tone(chorale_x[slice_counter],)
                    chord_class = get_chord_tone(chord, output_dim)
                    #chord_class = get_non_chord_tone(chorale_x[slice_counter], chord_class, output_dim)
                    chord_class = get_non_chord_tone_4(chorale_x[slice_counter], chord_class, output_dim, f_non)

                    slice_counter += 1
                    if(counter2 == 1):
                        y = np.concatenate((y, chord_class))
                    else:
                        y = np.vstack((y, chord_class))
            print('slices of output: ', slice_counter, "slices of input", slice_input)
            if abs(slice_counter - slice_input) >= 1 and slice_counter != 0:
                print('asdasd')
            # save x, y into file as "ground truth"

    # add zero to the matrix, so that it can be divided by 50
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    yy = [0] * outputdim
    yy[-1] = 1  # there should be a chord for these artificial slices, chord is 'other'
    print('yy:', yy)
    '''while(x.shape[0] % 50 !=0):
        x = np.vstack((x, [0] * inputdim))
        y = np.vstack((y, yy))'''
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    np.savetxt('.\\data_for_ML\\' +sign + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class_New_annotation.txt', x, fmt = '%.1e')
    np.savetxt('.\\data_for_ML\\' +sign + '_y_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class_New_annotation.txt', y, fmt = '%.1e')


def determine_middle_name(augmentation, source):
    '''
    Determine the file name of whether using augmentation, pitch or pitch-class and melodic or harmonic
    :param augmentation:
    :param pitch:
    :param source:
    :return:
    '''
    if (source == 'melodic'):
        music21 = ''
    else:  # we are gonna use maximally harmonic of music21 chordify module
        music21 = 'music21'

    if (augmentation == 'Y'):
        keys = '12keys'
    else:
        keys = 'keyC'
    return keys, music21
def generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1, output, f2, sign, predict, augmentation, pitch):
    """
    The only difference with "generate_data_windowing_non_chord_tone"
    :param counter1:
    :param counter2:
    :param string:
    :param string1:
    :param string2:
    :param x:
    :param y:
    :param inputdim:
    :param outputdim:
    :param windowsize:
    :param counter:
    :param countermin:
    :return:
    """
    file_counter = 0
    slice_counter = 0
    fn_total = []
    keys, music21 = determine_middle_name(augmentation, sign)
    if (predict == 'N'):
        number = 147
    else:
        number = 6
    if not (os.path.isfile('.\\data_for_ML\\' +sign + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_'+ pitch + '_New_annotation_' + keys +'_' +music21+'_' + str(number) + '.txt')):

        for id, fn in enumerate(os.listdir(input)):
            #print(fn)
            #if fn.find( 'KB') != -1 and fn[-4:] == f1 and fn.find('130') == -1 and fn.find('133') == -1 and fn.find('19') != -1:  # only look for the transposed one
            if(predict == 'N'):
                if fn.find('KB') != -1 and fn[-4:] == f1 and fn.find('130') == -1 and fn.find('133') == -1 and fn.find('340') == -1 and fn.find('358') == -1 and fn.find('362') == -1 and fn.find('003') == -1 and fn.find('008') == -1 and fn.find('014') == -1 :  # 001 173 371 as the test data
                #if(fn.find('134') != -1 and fn.find('g#') != -1 ):
                    if(augmentation != 'Y'):  # Don't want data augmentation in 12 keys
                        if(fn.find('cKE') != -1):  # only wants key c
                            fn_total.append(fn)
                    else:
                        fn_total.append(fn)
            else:
                if fn.find('KB') != -1:
                    if fn.find('340') != -1 or fn.find('358') != -1 or fn.find('362') != -1 or fn.find('003') != -1 or fn.find('008') != -1 or fn.find('014') != -1:
                        if (augmentation != 'Y'):
                            if (fn.find('cKE') != -1):
                                fn_total.append(fn)
                        else:
                            fn_total.append(fn)
        if(predict == 'N'):
            shuffle(fn_total)
        print (fn_total)
        #input('?')

        for id, fn in enumerate(fn_total):
                print(fn)
                ptr = fn.find('chor')
                chorale_x = []
                if (os.path.isfile(output + fn[:ptr] + 'translated_' + fn[-7:-4] + sign + f2)):
                    f = open(output + fn[:ptr] + 'translated_' + fn[-7:-4] + sign + f2,'r')
                    f_non = open(output + fn[:ptr] + 'non_chord_tone_' + music21 + '_'+ sign + fn[-7:-4] + f2,'w')
                else:
                    continue  # skip the file which does not have chord labels
                file_counter += 1
                s = converter.parse(input + fn)
                sChords = s.chordify()
                slice_input = 0
                #print(slice_input)
                #length = len(sChords)
                thisChordAll = []
                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    thisChordAll.append(thisChord)
                    slice_input += 1
                    counter1 += 1
                    slice_counter += 1
                    pitchClass = [0] * inputdim
                    #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                    pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                    pc_counter = 0
                    for ii in pitchClass:
                        if ii == 1:
                            pc_counter +=1
                    if (pc_counter > 4):
                        print("pc is greate than 4!~")
                    counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                    #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)  # add voice leading (or not)
                    #(thisChord.pitchClasses)
                    pitchClass = add_beat_into(pitchClass, thisChord.beatStr)  # add on/off beat info
                    if(i == 0):
                        chorale_x = np.concatenate((chorale_x, pitchClass))
                    else:
                        chorale_x = np.vstack((chorale_x, pitchClass))
                chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
                if(file_counter == 1):
                    x = chorale_x_window
                else:
                    x = np.concatenate((x, chorale_x_window))
                slice_counter = 0  # remember what slice in order to get the pitch class info
                for line in f.readlines():
                    line = get_chord_line(line, sign)

                    for chord in line.split():
                        if(chord.find('g]') != -1):
                            print(fn)
                            input('wtf is that?')
                        counter2 += 1
                        #chord_class = [0] * outputdim
                        #chord_class = y_non_chord_tone(chord, chord_class, list_of_chords)
                        #chord_class = get_non_chord_tone(chorale_x[slice_counter],)
                        chord_class = get_chord_tone(chord, outputdim)
                        #chord_class = get_non_chord_tone(chorale_x[slice_counter], chord_class, output_dim)
                        if(sign == 'melodic'):
                            chord_class = get_non_chord_tone_4(chorale_x[slice_counter], chord_class, outputdim, f_non)
                        else:
                            chord_class = get_non_chord_tone_4_music21(chorale_x[slice_counter], chord_class, f_non, thisChordAll[slice_counter])
                        slice_counter += 1
                        if(counter2 == 1):
                            y = np.concatenate((y, chord_class))
                        else:
                            y = np.vstack((y, chord_class))
                print('slices of output: ', slice_counter, "slices of input", slice_input)
                if abs(slice_counter - slice_input) >= 1 and slice_counter != 0:
                    print('asdasd')
                # save x, y into file as "ground truth"

        # add zero to the matrix, so that it can be divided by 50
        print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
        print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
        yy = [0] * outputdim
        yy[-1] = 1  # there should be a chord for these artificial slices, chord is 'other'
        print('yy:', yy)
        '''while(x.shape[0] % 50 !=0):
            x = np.vstack((x, [0] * inputdim))
            y = np.vstack((y, yy))'''
        print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
        #np.savetxt('.//data_for_ML//' + sign + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class_New_annotation_12keys_150.txt', x, fmt = '%.1e')


        np.savetxt('.\\data_for_ML\\' +sign + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_'+ pitch + '_New_annotation_' + keys +'_' +music21+'_' + str(number) + '.txt', x, fmt='%.1e')
        np.savetxt('.\\data_for_ML\\' +sign + '_y_windowing_' + str(windowsize) + 'y4_non-chord_tone_'+ pitch + '_New_annotation_' + keys +'_' +music21+'_' + str(number) + '.txt', y, fmt = '%.1e')


if __name__ == "__main__":
    counter = 0
    counterMin = 60
    # Get input features
    sign = '0'#input("do you want inversions or not? 1: yes, 0: no")
    output_dim =  '12'#input('how many kinds of chords do you want to calculate?')
    window_size = '0'#int(input('how big window?'))
    '''sign = 0
    output_dim = 30
    window_size = 1'''
    output_dim = int(output_dim)
    input_dim = 12
    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if (file_name[:6] == 'transl'):
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line = get_chord_line(line, sign)
                print(line)
                dic = calculate_freq(dic, line)
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    list_of_chords = []
    for i, word in enumerate(li):
        if(i == output_dim - 1):  # the last one is 'others'
            break
        list_of_chords.append(word[0])
    print (list_of_chords)  # Get the top 35 chord freq
    # Get the on/off beat info
    # Get the encodings for input
    counter1 = 0  # record the number of salami slices of poly
    counter2 = 0  # record the number of salami slices of chords
    input = '.\\bach-371-chorales-master-kern\\kern\\'
    output = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Melodic\\'
    f1 = '.xml'
    f2 = '.txt'
    generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, input_dim, output_dim, 2,
                                           counter, counterMin, input, f1, output, f2, 'melodic', 'Y')  # Y means predict the result for 6 chorales, no shuffle!





