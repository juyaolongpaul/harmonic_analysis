import unittest
import os
from music21 import *
from transpose_to_C_chords import change_length
from transpose_to_C_chords import get_displacement
from translate_output import is_next_a_chord
from translate_output import remove_candidates
from translate_output import only_one_nonchord_tone
from translate_output import replace_non_chord_tone_with_following_chord
from translate_output import translate_chord_line
'''c1=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b',]
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
                tonality.append(key_tonic + k.mode)'''
class SimpleTest(unittest.TestCase):
    def test_is_next_a_chord(self):
        test1 = ['[a','b','c]']
        test2 = ['[a', 'b']
        test3 = ['[a', 'b', '[c]']
        test4 = ['[a]', 'b', '[c']
        self.assertFalse(is_next_a_chord(test1,1))
        self.assertTrue(is_next_a_chord(test2, 1))
        self.assertTrue(is_next_a_chord(test3, 1))
        self.assertTrue(is_next_a_chord(test4, 1))
    def test_remove_candidates(self):
        test1 = ['am', 'g7', '([f', 'a]', 'dm/f)']
        test2 = ['([f', 'a]', 'dm/f)','am', 'g7']
        test3 = ['am', '([f', 'a]', 'dm/f)', 'g7']
        test4 = ['am', '([f', 'a]', 'dm/f)', 'g7']
        test5 = ['am', '(dm/f', '[f', 'a])', 'g7']
        test6 = ['am', '(dm/f', '[f', 'a]', '[e])', 'g7']
        test7 = ['am', '([f', 'a]', 'dm/f', '[e])', 'g7']
        self.assertEqual(remove_candidates(test1), ['am', 'g7', 'dm/f'])
        self.assertEqual(remove_candidates(test2), ['dm/f','am', 'g7'])
        self.assertEqual(remove_candidates(test3), ['am', 'dm/f', 'g7'])
        self.assertEqual(remove_candidates(test4), ['am', 'dm/f', 'g7'])
        self.assertEqual(remove_candidates(test5), ['am', 'dm/f', 'g7'])
        self.assertEqual(remove_candidates(test6), ['am', 'dm/f', 'g7'])
        self.assertEqual(remove_candidates(test7), ['am', 'dm/f', 'g7'])
    def test_only_one_nonchord_tone(self):
        test1 = ['[b]', '[b', 'g#]', 'am', '[d]', 'e', 'f#7#']
        test2 = ['[c#', 'e]', '[d', 'e]', 'em', '[g]']
        test3 = ['[b]', '[a]', '[c]', 'e', 'am', 'f#掳', 'g#掳']
        self.assertEqual(only_one_nonchord_tone(test1), ['[b]', '[b]', 'am', '[d]', 'e', 'f#7#'])
        self.assertEqual(only_one_nonchord_tone(test2), ['[c#]', '[d]', 'em', '[g]'])
        self.assertEqual(only_one_nonchord_tone(test3), ['[b]', '[a]', '[c]', 'e', 'am', 'f#掳', 'g#掳'])
    def test_replace_non_chord_tone_with_following_chord(self):
        test1 = ['[b]', '[a]', '[c]', 'e', 'am', 'f#掳', 'g#掳']
        self.assertEqual(replace_non_chord_tone_with_following_chord(test1,0), ['e', 'e', 'e', 'e', 'am', 'f#掳', 'g#掳'])
    def test_translate_chord_line(self):

if __name__ == "__main__":
    #test_change_length(c1, c2)
    #test_get_displacement()
    unittest.main()