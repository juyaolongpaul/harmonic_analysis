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
# Now there are only 32 cases, and the top 85% chords is actually 29 kinds
import re
import os
from music21 import *
#replace = '-*!{}\n'
replace = '{}\n'

cwd = '.\\bach_chorales_scores\\original_midi+PDF\\'
def replace_non_chord_tone_with_following_chord(chordcandidate,i):

        j = i + 1
        while (is_next_a_chord(chordcandidate, j) == False):
            j += 1
        if (j - i >= 3):
            print('alert!')
        for k in range(i, j):
            chordcandidate[k] = chordcandidate[j]
        return chordcandidate
def remove_candidates(chordcandidate):
    """
    Remove (), only keep the chord
    :param chordcandidate:
    :return:
    """
    for i, chord in enumerate(chordcandidate):
        if chord[0] == '(':
            print (chordcandidate)
            if chord[1] == '[':

                j = i
                while chordcandidate[j][-1] != ']':
                    j += 1
                for k in range(i, j + 1):
                    del chordcandidate[i] # delete all ones within []
                if(chordcandidate[i][0] != '['):  # a chord, delete the rest
                    if(chordcandidate[i][-1] == ')'): # only one chord, keep it
                        chordcandidate[i] = chordcandidate[i][:-1]
                    else:  # multiple chords, only keep the first one
                        j = i
                        while chordcandidate[j][-1] != ')':
                            j += 1
                        for k in range(i+1, j+1):
                            del chordcandidate[i+1]  # this section should ve been into a function!




                if(chordcandidate[i][-1] == ')'):
                    chordcandidate[i] = chordcandidate[i][:-1]
            else:  # first is the chord we need, delete the rest!

                chordcandidate[i] = chordcandidate[i][1:]  # remove (

                j = i
                while chordcandidate[j][-1] != ')':
                    j += 1
                for k in range(i+1, j+1):
                    del chordcandidate[i+1]  # this section should ve been into a function!
    return chordcandidate
def is_next_a_chord(chordcandidate, i):
    """
    Make sure whether the next element is a chord, can deal with [b, a, c] (c, [x x x]) this kidn of case now
    I do not think there can be
    :param chordcandidate:
    :param i:
    :param sign:
    :return:
    """
    if chordcandidate[i].find('[') == -1 and chordcandidate[i].find(']') == -1 and chordcandidate[i].find('(') == -1 and chordcandidate[i].find(')') == -1:
        if(len(chordcandidate) == i+1):  # there is not i+1, definately a chord
            return True
        elif(chordcandidate[i+1][-1] == ']' and chordcandidate[i+1][0] != '[' and chordcandidate[i-1][0] == '[' and chordcandidate[i-1][-1] != ']'):  # [x,x,x] case, not a chord
            return False
        else:
            return True  # other cases it is a chord!
    else:
        return False


def only_one_nonchord_tone(chord_candidate):
    """
    Remove the following non0-chord tone element, starting from ptr where it has [
    :param i:
    :return:
    """
    for i, chord in enumerate(chord_candidate):
        if(chord[0] == '['):
            if chord[-1] != ']':  # remove the rest of non-chord tones, only keep the first one
                        j = i
                        while chord_candidate[j][-1] != ']':
                            j += 1

                        for k in range(i+1, j+1):
                            #print(chord_candidate[i+1])
                            del chord_candidate[i+1] # do not use k, since the dimension will reduce!!!
                        chord_candidate[i] = chord + ']'

    return chord_candidate

def translate_chord_line(num_of_salami_poly, num_of_salami_chord, line, replace, beat_poly, bad):
    """
    Remove the sign in replace, and deal with non-chord tone and multiply answers
    :param line: the original chord line
    :param replace: the signs need to be removed
    :return:
    """
    chord_candidate = line.split()
    line = ''
    for chord in chord_candidate:
        ptr = chord.find('*')
        print('current chord' + chord)
        if (ptr != -1):
            print('current chord is:' + chord)
            # input('* found is chord!')
            temp = chord[:ptr]
            for i in range(int(chord[ptr+1])-1):
                temp = temp + ' ' + chord[:ptr]
                if(int(chord[ptr+1])>=3):
                    print('debug')
            chord = temp
        line += chord + ' '
    line = line[:-1]
    '''if(line.find('*2') != -1):
        print(line)
        print(fn)
        input('what does this symbol mean?')'''
    #line = re.sub(r'/\w+', '', line) # remove inversion
    for letter in replace:
        line = line.replace(letter, '')
    chord_candidate = line.split()
    #chord_candidate = ['[b]', '[b', 'g#]', 'am', '[d]', 'e', 'f#7#']
    #chord_candidate = ['[b', 'd]', 'am', '[b]', '[a]', 'e', 'am', 'f#掳', 'g#掳']
    #chord_candidate = ['[c#', 'e]', '[d', 'e]', 'em', '[g]']
    #chord_candidate = ['[b]', '[a]', '[c]', 'e', 'am', 'f#掳', 'g#掳']
    print('original one: ')
    print(chord_candidate)
    chord_candidate_original = chord_candidate
    chord_candidate = remove_candidates(chord_candidate)
    chord_candidate = only_one_nonchord_tone(chord_candidate)
    for i, chord in enumerate(chord_candidate):
        if(chord[0] != '(' and chord[0] != '['):  # just a chord
            num_of_salami_poly += 1
        elif chord[0] == '[':
            num_of_salami_poly += 1
            #chord_candidate= only_one_nonchord_tone(i, chord_candidate)
            if i != 0:  # replaced by the previous chord, or the following chord
                if(len(beat_poly[num_of_salami_poly - 1]) == 1):  # on beat, replaced by the following chord
                    if(len(chord_candidate) > i + 1):
                        chord_candidate = replace_non_chord_tone_with_following_chord(chord_candidate, i)  # this needs to be correct, otherwise the salami slices will be wrong!
                    else:
                        chord_candidate[i] = chord_candidate[i - 1]  # compromise
                        print(fn)
                        #input('current chord' + chord_candidate[i] + 'previous chord' + chord_candidate[i-1] )
                        bad += 1
                else:
                    chord_candidate[i] = chord_candidate[i - 1]   # off beat, replaced by the previous chord
                if chord_candidate[i-1][0] == '[':
                    print("non-chord tone preceded by another one???")
                    #bad += 1

            else:
                if (len(chord_candidate) > i + 1):
                    chord_candidate = replace_non_chord_tone_with_following_chord(chord_candidate, i)
                # begin with a non-chord tone
                if chord_candidate[i+1][0] == '[':
                    input("non-chord tone which begins the measure is followed by another one???")

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
            if ((os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop''')) or
                (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'transposed_' + fn[0:3] + '.pop.not'''))):

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
            else:
                print('no file is opened')
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line, num_of_salami_poly, num_of_salami_chord, bad = translate_chord_line(num_of_salami_poly, num_of_salami_chord, line, replace, beat_poly, bad)
                print (line)
                if ('[' in line) or (']' in line) or ('(' in line) or (')' in line):
                    print(fn)
                    print(line)
                    print('what do you think?')
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
