from music21 import *
import os
from counter_chord_frequency import *
format = ['mid']
cwd = '.\\bach_chorales_scores\\original_midi+PDF\\'

def fill_in_pitch_class(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        pitchclass[i] = 1
    return pitchclass



if __name__ == "__main__":
    # Get input features
    for fn in os.listdir(cwd):
        print(fn)
        if fn[-3:] == 'mid':

            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            print(len(sChords.notes))
            for thisChord in sChords.recurse().getElementsByClass('Chord'):
                pitchClass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pitchClass = fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                print(thisChord.pitchClasses)
                print(pitchClass)

            input('asd')
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
