# Translate output encodings to mere chord symbols.
# []: non-chord tone is replaced by whatever chord before it, if a measure begin with [],
# it is replaced by the following chord
# (): only the first chord of the () will be considered as the answer
# {}: is ignored
# the transformed output should have the same number of slices with the original one!
import re
import os
replace = '-*!{}\n'


def translate_chord_line(line, replace):
    """
    Remove the sign in replace, and deal with non-chord tone and multiply answers
    :param line: the original chord line
    :param replace: the signs need to be removed
    :return:
    """
    for letter in replace:
        line = line.replace(letter, '')
    line = re.sub(r'/\w+', '', line)
    chord_candidate = line.split()
    print('original one: ')
    print(chord_candidate)
    chord_candidate_original = chord_candidate
    for i, chord in enumerate(chord_candidate):
        if chord[0] == '[':
            if chord[-1] != ']':  # remove the rest of non-chord tones, only keep the first one
                j = i
                while chord_candidate[j][-1] != ']':
                    j += 1
                for k in range(i+1, j+1):
                    chord_candidate.remove(chord_candidate[i+1]) # do not use k, since the dimension will reduce!!!
            if i != 0:  # replaced by the previous chord
                chord_candidate[i] = chord_candidate[i-1]
                if chord_candidate[i-1][0] == '[':
                    input("non-chord tone preceded by another one???")
            else:
                j = i
                while chord_candidate[j][0] == '[':
                    j += 1
                for k in range(i, j):
                    chord_candidate[k] = chord_candidate[j]
                if chord_candidate[j][0] == '[':
                    input("non-chord tone which begins the measure is followed by another one???")
        if chord[0] == '(':
            if chord[1] == '[':
                if chord[-1] != ']':  # remove the rest of non-chord tones, only keep the first one
                    j = i
                    while chord_candidate[j][-1] != ']':
                        j += 1
                    for k in range(i, j + 1):
                        chord_candidate.remove(chord_candidate[i]) # delete all ones within []
                else:
                    chord_candidate.remove(chord_candidate[i])
                chord_candidate[i] = chord_candidate[i][:-1]
            else:  # first is the chord we need, delete the rest!
                chord_candidate[i] = chord_candidate[i][1:]
                if chord[-1] != ']':
                    j = i
                    while chord_candidate[j][-1] != ')':
                        j += 1
                    for k in range(i+1, j+1):
                        chord_candidate.remove(chord_candidate[i+1]) #  this section should ve been into a function!
    print('translated one: ')
    print(chord_candidate)
    if len(chord_candidate) != len(chord_candidate_original):
        input("translated one miss something...")
    #for chord in line.split():
    line = ''
    for chord in chord_candidate:
        line += chord + ' '
    line = line[:-1] # remove the last space
    return line


if __name__ == "__main__":

    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if file_name[:5] == 'trans':
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line = translate_chord_line(line, replace)
                print (line)
                if ('[' in line) or (']' in line) or ('(' in line) or (')' in line):
                    input('[]() still exists, bug!')  # exmaine