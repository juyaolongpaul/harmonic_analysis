from music21 import *
import os
import re
import numpy as np
dic = {}
from counter_chord_frequency import *
from adding_window_one_hot import adding_window_one_hot
from test_musicxml_gt import get_chord_tone
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
    if(sign == '0'):
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
    pitchclass = [0] * 12  # calculate by pitch_distribution
    for i in list:
        pitchclass[(i.midi-24) % 12] = 1  # the lowest is 28, compressed
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


def get_non_chord_tone_4(x,y,outputdim):
    """
    Take out chord tones, only leave with non-chord tone

    :param x:
    :param y:
    :return:
    """
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    for i in range(12):
        if(x[i] == 1):  # non-chord tone
            yyptr += 1
            if(y[i%12] == 0):
                yy[yyptr] = 1

    return yy



def generate_data_windowing_non_chord_tone(counter1, counter2, string, string1, string2, x, y, inputdim, outputdim, windowsize, counter, countermin):
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
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

                counter1 += 1
                slice_counter += 1
                pitchClass = [0] * inputdim
                #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                #pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
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
                    chord_class = get_non_chord_tone_4(chorale_x[slice_counter], chord_class, output_dim)
                    slice_counter += 1
                    if(counter2 == 1):
                        y = np.concatenate((y, chord_class))
                    else:
                        y = np.vstack((y, chord_class))

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
    np.savetxt(string + string1 + string2 + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone.txt', x, fmt = '%.1e')
    np.savetxt(string + string1 + string2 + '_y_windowing_' + str(windowsize) + 'y4_non-chord_tone.txt', y, fmt = '%.1e')

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



if __name__ == "__main__":
    counter = 0
    counterMin = 60
    # Get input features
    sign = '1'#input("do you want inversions or not? 1: yes, 0: no")
    output_dim =  '12'#input('how many kinds of chords do you want to calculate?')
    window_size = '1'#int(input('how big window?'))
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
    generate_data_windowing_non_chord_tone(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 1, counter, counterMin)
    #generate_data_windowing_non_chord_tone(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 2,
                                           #counter, counterMin)
    #generate_data_windowing_non_chord_tone(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 3,
                                           #counter, counterMin)
    #generate_data_windowing(counter1, counter2, 'valid', 'valid', 'valid', x, y, input_dim, output_dim, 1, counter,
                            #counterMin)
    #generate_data_windowing_no_y(counter1, 'train', 'valid', 'test', x, input_dim, 1, counter, counterMin)
    #generate_data_windowing(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 2)
    #generate_data_windowing(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 3)
    #generate_data_windowing(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 4)
    #generate_data_windowing(counter1, counter2, 'train', 'valid', 'test', x, y, input_dim, output_dim, 5)
    #generate_data_windowing(counter1, counter2, 'valid', 'valid', 'valid', x, y, input_dim, output_dim, window_size)
    #generate_data_windowing(counter1, counter2, 'test', 'test', 'test', x, y, input_dim, output_dim, window_size)
    #generate_data_windowing(counter1, counter2, 'valid', x, y, input_dim, output_dim, window_size)
    #generate_data_windowing(counter1, counter2, 'test', x, y, input_dim, output_dim, window_size)
    #generate_data_CV(counter1, counter2, 'train', 'valid', x, y, input_dim, output_dim)
    # Get output labels




