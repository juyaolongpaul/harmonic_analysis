# Translate output encodings to mere chord symbols.
# []: non-chord tone is replaced by whatever chord before it, if a measure begin with [],
# it is replaced by the following chord
# (): only the first chord of the () will be considered as the answer
# {}: is ignored
# the transformed output should have the same number of slices with the original one!
# For the ones that needs to be replaced by the following chord which is not a chord, it is replaced by non-chord tone, this number
# is really small, 116/30000, and when it calculates the chord frequency, it will be of top 35, and it will be filtered out,
# and the encodings will be all zero
# However, the method above is not usable,
# since the non-chord tone will mess up with the number of salami slices, so about 70 files are not usable,
# they are replaced by the previous chord (116 cases)
import re
import os
from music21 import *
replace = '-*!{}\n'
cwd = '.\\bach_chorales_scores\\original_midi+PDF\\'

def remove_chord_tone(i, chord_candidate):
    """
    Remove the following non0-chord tone element, starting from ptr where it has [
    :param i:
    :return:
    """
    if chord_candidate[i][-1] != ']':  # remove the rest of non-chord tones, only keep the first one
                j = i
                while chord_candidate[j][-1] != ']':
                    j += 1

                for k in range(i+1, j+1):
                    #print(chord_candidate[i+1])
                    del chord_candidate[i+1] # do not use k, since the dimension will reduce!!!

    return chord_candidate

def translate_chord_line(num_of_salami_poly, num_of_salami_chord, line, replace, beat_poly, bad):
    """
    Remove the sign in replace, and deal with non-chord tone and multiply answers
    :param line: the original chord line
    :param replace: the signs need to be removed
    :return:
    """
    for letter in replace:
        line = line.replace(letter, '')
    #line = re.sub(r'/\w+', '', line) # remove inversion
    chord_candidate = line.split()
    #chord_candidate = ['[b]', '[b', 'g#]', 'am', '[d]', 'e', 'f#7#']
    #chord_candidate = ['[b', 'd]', 'am', '[b]', '[a]', 'e', 'am', 'f#掳', 'g#掳']
    #chord_candidate = ['[c#', 'e]', '[d', 'e]', 'em', '[g]']
    #chord_candidate = ['[b]', '[a]', '[c]', 'e', 'am', 'f#掳', 'g#掳']
    print('original one: ')
    print(chord_candidate)
    chord_candidate_original = chord_candidate
    for i, chord in enumerate(chord_candidate):
        if(chord[0] != '(' and chord[0] != '['):  # just a chord
            num_of_salami_poly += 1
        elif chord[0] == '[':
            num_of_salami_poly += 1
            chord_candidate= remove_chord_tone(i, chord_candidate)
            if i != 0:  # replaced by the previous chord, or the following chord
                if(len(beat_poly[num_of_salami_poly - 1]) == 1):  # on beat, replaced by the following chord
                    if(len(chord_candidate) > i + 1):
                        if(chord_candidate[i + 1] != '(' and chord[0] != '['):
                            chord_candidate[i] = chord_candidate[i + 1]
                        else:
                            chord_candidate[i] = chord_candidate[i - 1]
                    else:
                        chord_candidate[i] = chord_candidate[i - 1]  # compromise
                else:
                    chord_candidate[i] = chord_candidate[i - 1]   # off beat, replaced by the previous chord
                if chord_candidate[i-1][0] == '[':
                    print("non-chord tone preceded by another one???")
                    bad += 1
            else:
                if(chord_candidate[i+1][0] == '['):
                    remove_chord_tone(i+1, chord_candidate)
                    if(chord_candidate[i+2][0] == '['):

                        remove_chord_tone(i+2, chord_candidate)
                        if(chord_candidate[i+3][0] == '['):
                            input('[][][][] in a row???')  # better way to do this recursive problem???

                        elif(chord_candidate[i+3][0] == '('):  # need to select the chord from this () thing
                            if(chord_candidate[i+3][1] != '['):  # this is the chord we need
                                chord_candidate[i+2] = chord_candidate[i+3][1:]
                                chord_candidate[i+1] = chord_candidate[i+2][1:]
                                chord_candidate[i] = chord_candidate[i+1]
                            else:
                                input("we need to deal with [] [] ([] thing!")

                        else:
                            chord_candidate[i+2] = chord_candidate[i+3]
                            chord_candidate[i+1] = chord_candidate[i+2]
                            chord_candidate[i] = chord_candidate[i+2]

                    elif(chord_candidate[i+2][0] == '('):  # need to select the chord from this () thing
                        if(chord_candidate[i+2][1] != '['):  # this is the chord we need
                            chord_candidate[i+1] = chord_candidate[i+2][1:]
                            chord_candidate[i] = chord_candidate[i+1]
                        else:
                            input("we need to deal with [] [] ([] thing!")

                    else:
                        chord_candidate[i+1] = chord_candidate[i+2]
                        chord_candidate[i] = chord_candidate[i+2]
                elif(chord_candidate[i+1][0] == '('):  # need to select the chord from this () thing
                        if(chord_candidate[i+1][1] != '['):  # this is the chord we need
                            chord_candidate[i] = chord_candidate[i+1][1:]
                        else:
                            input("we need to deal with [] [] ([] thing!")

                else:
                    chord_candidate[i] = chord_candidate[i+1]  # the beginning with the non-chord tone must be on beat, replaced by the following chord
                if chord_candidate[i+1][0] == '[':
                    input("non-chord tone which begins the measure is followed by another one???")
        elif chord[0] == '(':
            num_of_salami_poly += 1
            if chord[1] == '[':

                j = i
                while chord_candidate[j][-1] != ']':
                    j += 1
                for k in range(i, j + 1):
                    del chord_candidate[i] # delete all ones within []
                if(chord_candidate[i][0] != '['):  # a chord, delete the rest
                    if(chord_candidate[i][-1] == ')'): # only one chord, keep it
                        chord_candidate[i] = chord_candidate[i][:-1]
                    else:  # multiple chords, only keep the first one
                        j = i
                        while chord_candidate[j][-1] != ')':
                            j += 1
                        for k in range(i+1, j+1):
                            del chord_candidate[i+1]  # this section should ve been into a function!




                if(chord_candidate[i][-1] == ')'):
                    chord_candidate[i] = chord_candidate[i][:-1]
            else:  # first is the chord we need, delete the rest!

                chord_candidate[i] = chord_candidate[i][1:]  # remove (

                j = i
                while chord_candidate[j][-1] != ')':
                    j += 1
                for k in range(i+1, j+1):
                    del chord_candidate[i+1]  # this section should ve been into a function!

    print('translated one: ')

    print(chord_candidate)
    num_of_salami_chord += len(chord_candidate)
    print('num_of_salami_poly' + str(num_of_salami_poly))
    print('num_of_salami_chord' + str(num_of_salami_chord))
    if(num_of_salami_poly != num_of_salami_chord):
        input('salami number is wrong!')
    if len(chord_candidate) != len(chord_candidate_original):
        input("translated one miss something...")
    #for chord in line.split():
    line = ''
    for chord in chord_candidate:
        line += chord + ' '
    line = line[:-1] # remove the last space
    return line, num_of_salami_poly, num_of_salami_chord, bad


if __name__ == "__main__":
    bad = 0
    num_of_samples = 0
    for fn in os.listdir(cwd):
        if fn[-3:] == 'mid':
            print(fn)
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            print(len(sChords.notes))
            num_of_samples += len(sChords.notes)
            beat_poly = ['0'] * len(sChords.notes)
            num_of_salami_poly = 0
            num_of_salami_chord = 0
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                beat_poly[i] = thisChord.beatStr # the ptr is not accurate, but the off-beat can still be detected
                if(len(thisChord.beatStr) != 1):
                    print('beat location is: ' + thisChord.beatStr)

            if (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop''')):
                f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop','r')
                file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop'
                fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + 'transposed_' + fn[0:3] + '.pop', 'w')
            elif (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop.not''')):
                f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop.not','r')
                file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop.not'
                fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + 'transposed_' + fn[0:3] + '.pop.not', 'w')
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line, num_of_salami_poly, num_of_salami_chord, bad = translate_chord_line(num_of_salami_poly, num_of_salami_chord, line, replace, beat_poly, bad)
                print (line)
                if ('[' in line) or (']' in line) or ('(' in line) or (')' in line):
                    #input('[]() still exists, bug!')  # exmaine
                    line = re.sub(r'\[\w\]', '', line)
                    line = re.sub(r'\[\w\w]', '', line)
                    line = re.sub(r'\[\w\W]', '', line)

                    line = re.sub(r'\[\w \w]', '', line) # nesty exceptions
                    print('new line:' + line)
                    #input('what do you think?')
                    #pattern = re.compile(r'\w[+]')
                    #if(pattern):  # deal with  'c[d]' case
                print(line, end='\n', file=fnew)
    print('bad =' + str(bad))
    print ('number of samples = ' + str(num_of_samples))
