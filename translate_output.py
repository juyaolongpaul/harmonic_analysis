# Translate output encodings to mere chord symbols.
# []: non-chord tone is replaced by whatever chord before it, if a measure begin with [],
# it is replaced by the following chord
# (): only the first chord of the () will be considered as the answer
# {}: is ignored
# the transformed output should have the same number of slices with the original one!
import re
import os
replace = '-*!{}\n'


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

def translate_chord_line(line, replace):
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
        if chord[0] == '[':

            chord_candidate= remove_chord_tone(i, chord_candidate)
            if i != 0:  # replaced by the previous chord
                chord_candidate[i] = chord_candidate[i-1]
                if chord_candidate[i-1][0] == '[':
                    input("non-chord tone preceded by another one???")
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
                    chord_candidate[i] = chord_candidate[i+1]
                if chord_candidate[i+1][0] == '[':
                    input("non-chord tone which begins the measure is followed by another one???")
        if chord[0] == '(':
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
        if file_name[:6] == 'transp':
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\'+ 'translated_' + file_name, 'w')
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
