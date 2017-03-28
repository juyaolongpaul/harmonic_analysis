from music21 import *
import os

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


for fn in os.listdir(cwd):

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
