import unittest
import os
from music21 import *
from transpose_to_C_chords import change_length
from transpose_to_C_chords import get_displacement
c1=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b',]
c2=['c','db','d','eb','e','f','gb','g','ab','a','bb','b',]
def test_change_length(c1,c2):
    """

    :param c1: original pitch_class
    :param c2: transposed pitch_class
    :return: the displacement
    """
    mark = 0
    for i in c1:
        for j in c2:

            mark = change_length(i, j, mark)
            mark_test = len(j) - len(i)
            if(mark_test != mark):
                print("Error")
                print("The original pitch class: " + i)
                print("The transposed pitch class: " + j)
                print("The displacement is: " + str(mark))
                print("The real displacement is: " + str(mark_test))
def test_get_displacement():
    """
    Manually examine.
    :return:
    """
    tonality = []
    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):

            if file_name[-3:] == 'pop' or file_name[-3:] == 'not':
                #if(file_name[:3] != '369'):
                    #continue
                ptr = file_name.find('.')
                s = converter.parse(os.getcwd() + '\\bach_chorales_scores\\original_midi+PDF\\' + file_name[:ptr]+'.mid')
                k = s.analyze('key')
                ptr = k.name.find(' ')
                key_tonic = k.name[:ptr]
                key_tonic = key_tonic.lower()
                key_tonic = key_tonic.replace('-', 'b')
                displacement = get_displacement(k)
                if(key_tonic + k.mode not in tonality):
                    print("The key is: " + key_tonic + k.mode + " The displacement is: " + str(displacement))
                tonality.append(key_tonic + k.mode)
if __name__ == "__main__":
    #test_change_length(c1, c2)
    test_get_displacement()