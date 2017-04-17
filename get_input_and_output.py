from music21 import *
import os
import re
import numpy as np
dic = {}
from counter_chord_frequency import *
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
    #for letter in replace:
        #line = line.replace(letter, '')
    if(sign == '0'):
        line = re.sub(r'/\w+', '', line)  # remove inversions
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
    :param list: The chord list with 35 top freq
    :return: the modified pitch class that need to store
    """
    for i, chord2 in enumerate(list):
        if(chord == chord2):

            chordclass[i] = 1
            break
    return chordclass


def fill_in_pitch_class(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        pitchclass[i] = 1
    return pitchclass


def generate_data(counter1, counter2, string, x, y):
    for fn in os.listdir(cwd):
        print(fn)
        if fn[-3:] == 'mid':
            if (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop','r')

            elif (os.path.isfile('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\' + string + '\\translated_transposed_' + fn[0:3] + '.pop.not','r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            for line in f.readlines():
                line = get_chord_line(line, sign)
                for chord in line.split():
                    counter2 += 1
                    chord_class = [0] * 35
                    chord_class = fill_in_chord_class(chord, chord_class, list_of_chords)
                    if(counter2 == 1):
                        y = np.concatenate((y, chord_class))
                    else:
                        y = np.vstack((y, chord_class))
            for thisChord in sChords.recurse().getElementsByClass('Chord'):
                counter1 += 1
                pitchClass = [0] * 12
                pitchClass = fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                print(thisChord.pitchClasses)
                print(pitchClass)
                if(counter1 == 1):
                    x = np.concatenate((x, pitchClass))
                else:
                    x = np.vstack((x, pitchClass))
    # add zero to the matrix, so that it can be divided by 50
    print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
    while(x.shape[0] % 50 !=0):
        x = np.vstack((x, [0] * 12))
        y = np.vstack((y, [0] * 35))
    print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
    np.savetxt(string + '_x.txt', x, fmt = '%.1e')
    np.savetxt(string + '_y.txt', y, fmt = '%.1e')


if __name__ == "__main__":
    # Get input features
    sign = input("do you want inversions or not? 1: yes, 0: no")
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
        if(i == 35):
            break
        list_of_chords.append(word[0])
    print (list_of_chords)  # Get the top 35 chord freq

    # Get the encodings for input
    counter1 = 0  # record the number of salami slices of poly
    counter2 = 0  # record the number of salami slices of chords
    generate_data(counter1, counter2, 'train', x, y)
    generate_data(counter1, counter2, 'valid', x, y)
    generate_data(counter1, counter2, 'test', x, y)
    # Get output labels



    '''for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if(file_name[:5] == 'trans'):
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)
                line = get_chord_line(line, replace)
                print(line)
                dic = calculate_freq(dic, line)
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)  # sort it'''
