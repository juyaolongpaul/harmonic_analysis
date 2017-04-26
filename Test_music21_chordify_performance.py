from music21 import *
import os
import re
import numpy as np
dic = {}
from counter_chord_frequency import *
format = ['mid']
cwd = '.\\bach_chorales_scores\\transposed_MIDI\\'

def fill_in_pitch_class(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        pitchclass[i] = 1
    return pitchclass

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

def translate_gt_chord_quality(quality):


def translate_gt(chord):
    if(len(chord) == 2):
        if(chord[1] == 'b' or chord[1] == '#' ):  # flat or sharp
            root = chord[0] + chord[1]
        else:
            root = chord[0]
            quality1, quality2 = translate_gt_chord_quality(chord[1:])


def generate_vector(gtroot, gtquality1, gtquality2, chordifyroot, chordifyquality1, chordifyquality2):
    for fn in os.listdir(cwd):
        print(fn)
        if fn[-3:] == 'mid':
            if (os.path.isfile('.\\useful_chord_symbols\\translated_transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\useful_chord_symbols\\translated_transposed_' + fn[0:3] + '.pop', 'r')

            elif (
            os.path.isfile('.\\useful_chord_symbols\\translated_transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\translated_transposed_' + fn[0:3] + '.pop.not', 'r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            for line in f.readlines():
                line = get_chord_line(line, sign)
                for chord in line.split():
                    groundTruth.append(chord)

            for thisChord in sChords.recurse().getElementsByClass('Chord'):

                pitchClass = [0] * 12
                pitchClass = fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                print(thisChord.pitchClasses)
                print('chord is' + thisChord.pitchedCommonName)
                print('chord is: ' + thisChord._root.name + ' ' + thisChord.quality)


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
    gtRoot = []
    gtQuality1 = []
    gtQuality2 = []
    chordifyRoot = []
    chordifyQuality1 = []
    chordifyQuality2 = []

    generate_vector(gtRoot, gtQuality1, gtQuality2, chordifyRoot, chordifyQuality1, chordifyQuality2)
