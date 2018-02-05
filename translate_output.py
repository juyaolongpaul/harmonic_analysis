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
            if chordcandidate[k].find('[') != -1 and chordcandidate[k].find(']') != -1 and chordcandidate[k][0] != \
                    '(' and chordcandidate[k][0] != '[':
                ptr1 = chordcandidate[k].find('[')
                ptr2 = chordcandidate[k].find(']')
                chordcandidate[k] = chordcandidate[k][:ptr1] + chordcandidate[j] + chordcandidate[k][ptr2 + 1:]
            else:
                chordcandidate[k] = chordcandidate[j]
        return chordcandidate

def replace_non_chord_tone_with_previous_chord(chordcandidate, i):
    """
        This is the function to replace the non-chord tone in multiple interpretation
        :param chordcandidate:
        :param i:
        :return:
        """
    ptr1 = chordcandidate[i].find('[')
    ptr2 = chordcandidate[i].find(']')
    if ptr1 == -1 or ptr2 == -1:
        input('no non-chord tone in multiple interpretaions??')
    chordcandidate[i] = chordcandidate[i][:ptr1] + chordcandidate[i - 1] + chordcandidate[i][ptr2 + 1:]
    return chordcandidate

def replace_non_chord_tone_with_following_chord1(chordcandidate, i):
    """
    This is the function to replace the non-chord tone in multiple interpretation
    :param chordcandidate:
    :param i:
    :return:
    """
    ptr1 = chordcandidate[i].find('[')
    ptr2 = chordcandidate[i].find(']')
    if ptr1 == -1 or ptr2 == -1 :
        input('no non-chord tone in multiple interpretaions??')
    j = i + 1
    while (is_next_a_chord(chordcandidate, j) == False):
        j += 1
    if (j - i >= 3):
        print('alert!')
    for k in range(i, j):
        if chordcandidate[k].find('[') != -1 and chordcandidate[k].find(']') != -1 and chordcandidate[k][0] != \
                '(' and chordcandidate[k][0] != '[':
            ptr1 = chordcandidate[k].find('[')
            ptr2 = chordcandidate[k].find(']')
            chordcandidate[k] = chordcandidate[k][:ptr1] + chordcandidate[j] + chordcandidate[k][ptr2 + 1:]
        else:
            chordcandidate[k] = chordcandidate[j]
    return chordcandidate
def remove_candidates(chordcandidate, num_of_ambiguity):
    """
    Remove (), only keep the chord
    :param chordcandidate:
    :return:
    """
    for i, chord in enumerate(chordcandidate):
        if chord[0] == '(':
            num_of_ambiguity += 1
            print(chordcandidate)
            if chord[1] == '[':

                j = i
                while chordcandidate[j][-1] != ']':
                    j += 1
                for k in range(i, j + 1):
                    del chordcandidate[i]  # delete all ones within []

                if (chordcandidate[i][0] != '['):  # a chord, delete the rest
                    if (chordcandidate[i][-1] == ')'):  # only one chord, keep it
                        chordcandidate[i] = chordcandidate[i][:-1]
                    else:  # multiple chords, only keep the first one
                        j = i
                        while chordcandidate[j][-1] != ')':
                            j += 1
                        for k in range(i + 1, j + 1):
                            del chordcandidate[i + 1]  # this section should ve been into a function!

                if (chordcandidate[i][-1] == ')'):
                    chordcandidate[i] = chordcandidate[i][:-1]
            else:  # first is the chord we need, delete the rest!

                chordcandidate[i] = chordcandidate[i][1:]  # remove (

                j = i
                while chordcandidate[j][-1] != ')':
                    j += 1
                for k in range(i + 1, j + 1):
                    del chordcandidate[i + 1]  # this section should ve been into a function!
    return chordcandidate, num_of_ambiguity
def keep_candidates(chordcandidate, num_of_ambiguity):
    """
    Remove (), the chords and non-chord tones
    :param chordcandidate:
    :return:
    """
    for i, chord in enumerate(chordcandidate):
        multi_interpretation = [None] * 10
        num_of_interpretation = 0
        if chord[0] == '(':
            num_of_ambiguity += 1
            print (chordcandidate)
            if chord[1] == '[':

                j = i
                while chordcandidate[j][-1] != ']':
                    j += 1
                for k in range(i, j + 1):
                    if k == j:  # keep one non-chord tone for labeling ambiguity
                        multi_interpretation[num_of_interpretation] = chordcandidate [i]  # incorporate
                        # non-chord + interpretation.
                        num_of_interpretation += 1
                    del chordcandidate[i] # delete all ones within []
                if chordcandidate[i][0] != '[' :  # a chord, delete the rest
                    if chordcandidate[i][-1] == ')':  # only one chord, keep it
                        chordcandidate[i] = chordcandidate[i][:-1]
                    else:  # multiple chords, only keep the first one
                        j = i
                        while chordcandidate[j][-1] != ')':
                            j += 1
                        for k in range(i+1, j+1):
                            multi_interpretation[num_of_interpretation] = chordcandidate[i + 1]  # incorporate the
                            # multi-chord interpretation.
                            num_of_interpretation += 1
                            del chordcandidate[i+1]  # this section should ve been into a function!




                if(chordcandidate[i][-1] == ')'):
                    chordcandidate[i] = chordcandidate[i][:-1]
            else:  # first is the chord we need, delete the rest!

                chordcandidate[i] = chordcandidate[i][1:]  # remove (

                j = i
                while chordcandidate[j][-1] != ')':
                    j += 1
                for k in range(i+1, j+1):
                    multi_interpretation[num_of_interpretation] = chordcandidate[i + 1]  # incorporate the
                    # multi-chord interpretation.
                    num_of_interpretation += 1
                    del chordcandidate[i+1]  # this section should ve been into a function!
            print('debug multi')
            print('now the chord candidates:', chordcandidate, 'index of it:' , i)
            print('multi interpretations: ' , multi_interpretation)
            for ii in (0, num_of_interpretation):
                if multi_interpretation[ii] is not None and multi_interpretation[ii].find('[') != -1 :  # only keep
                    #  one non-chord tone
                    jj = ii
                    while multi_interpretation[jj] is not None and multi_interpretation[jj].find(']') == -1 :
                        jj += 1
                    for k in range(ii, jj):
                        del multi_interpretation[ii]  # delete all ones within []
            print('multi interpretations (modified): ', multi_interpretation)
            for ii, item in enumerate(multi_interpretation):
                if item is not None:
                    if item.find(']') != -1 and item.find('[') == -1:  # make full []
                        item = '[' + item
                    item = item.replace(')', '')
                    item = item.replace('(', '')
                    chordcandidate[i] = chordcandidate[i] + ',' + item
            print('chord candidates with multi:', chordcandidate, 'index of it:', i)
            #input('?')
    return chordcandidate, num_of_ambiguity
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
def replace_with_chords(beat_poly, num_of_salami_poly, chord_candidate, i, bad):
    """

    :param beat_poly:
    :param num_of_salami_poly:
    :param chord_candidate:
    :param i:
    :param bad:
    :return:
    """
    if i != 0:  # replaced by the previous chord, or the following chord
        if (len(beat_poly[num_of_salami_poly - 1]) == 1):  # on beat, replaced by the following chord
            if  len(chord_candidate) > i + 1 :
                if chord_candidate[i].find('[') != -1 and chord_candidate[i].find(']') != -1 and chord_candidate[i][0] !=\
                        '(' and chord_candidate[i][0] != '[':
                    chord_candidate = replace_non_chord_tone_with_following_chord1(chord_candidate,i)
                else:
                    chord_candidate = replace_non_chord_tone_with_following_chord(chord_candidate,i)  # this needs to be correct, otherwise the salami slices will be wrong!
            else:
                chord_candidate[i] = chord_candidate[i - 1]  # compromise
                #print(fn)
                # input('current chord' + chord_candidate[i] + 'previous chord' + chord_candidate[i-1] )
                bad += 1
        else:
            if chord_candidate[i].find('[') != -1 and chord_candidate[i].find(']') != -1 and chord_candidate[i][0] != \
                    '(' and chord_candidate[i][0] != '[':
                chord_candidate = replace_non_chord_tone_with_previous_chord(chord_candidate, i)
            else:
                chord_candidate[i] = chord_candidate[i - 1]  # off beat, replaced by the previous chord
        if chord_candidate[i - 1][0] == '[':
            print("non-chord tone preceded by another one???")
            # bad += 1

    else:  # can only replaced with the following chord
        if (len(chord_candidate) > i + 1):
            if chord_candidate[i].find('[') != -1 and chord_candidate[i].find(']') != -1 and chord_candidate[i][0] != \
                    '(' and chord_candidate[i][0] != '[':
                chord_candidate = replace_non_chord_tone_with_following_chord1(chord_candidate, i)
            else:
                chord_candidate = replace_non_chord_tone_with_following_chord(chord_candidate, i)
        # begin with a non-chord tone
        if chord_candidate[i + 1][0] == '[':
            input("non-chord tone which begins the measure is followed by another one???")
    return beat_poly, num_of_salami_poly, chord_candidate, i, bad
def translate_chord_line(num_of_salami_poly, num_of_salami_chord, line, replace, beat_poly, bad, fn, num_of_ambiguity, multi):
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
    if(multi==0):
        chord_candidate, num_of_ambiguity = remove_candidates(chord_candidate, num_of_ambiguity)
    else:
        chord_candidate, num_of_ambiguity = keep_candidates(chord_candidate, num_of_ambiguity)
    chord_candidate = only_one_nonchord_tone(chord_candidate)
    for i, chord in enumerate(chord_candidate):  # replace non-chord tone with either the previous chord or the following chord
        if(chord[0] != '(' and chord[0] != '['):  # just a chord
            num_of_salami_poly += 1
            if chord.find('[') != -1 and chord.find(']') != -1 :  # replace non-chord into chord
                beat_poly, num_of_salami_poly, chord_candidate, i, bad = replace_with_chords(beat_poly,
                                                                                             num_of_salami_poly,
                                                                                             chord_candidate, i, bad)
        elif chord[0] == '[':
            num_of_salami_poly += 1
            #chord_candidate= only_one_nonchord_tone(i, chord_candidate)
            beat_poly, num_of_salami_poly, chord_candidate, i, bad = replace_with_chords(beat_poly, num_of_salami_poly, chord_candidate, i, bad)


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
    return line, num_of_salami_poly, num_of_salami_chord, bad, num_of_ambiguity


def annotation_to_txt():
    """
    A function that extract chord labels from musicxml to txt and translate them
    :return:

    """
    cwd_score = '.\\bach-371-chorales-master-kern\\kern\\'  # for new annotations, we have to use krn version,
    # since it does not equal to the original midi version completely and mid cannot be visualized.
    cwd_annotation = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\'
    cwd_annotation_m= '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Melodic\\'
    cwd_annotation_h = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Harmonic\\'
    #print(os.listdir(cwd_annotation))
    for fn in os.listdir(cwd_annotation):
        if (os.path.isfile(cwd_annotation_m + 'translated_' + fn[10:13] + 'melodic.txt')):  # if files are already there, jump out
            break
        if fn[-3:] == 'xml':
            original = []
            melodic = []
            harmonic = []
            print(fn)
            s = converter.parse(cwd_annotation + fn)
            sChords = s.parts[4]
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                #print(fn, i)

                if(len(thisChord.lyrics)== 5):
                    original.append(thisChord.lyrics[0].text)
                    melodic.append(thisChord.lyrics[3].text)
                    harmonic.append(thisChord.lyrics[4].text)
                elif(len(thisChord.lyrics) == 4):  # multi is missing
                    if len(thisChord.lyrics[1].text) == 3 and thisChord.lyrics[1].text.find('/') != -1 :  # confirm multi is missing
                        original.append(thisChord.lyrics[0].text)
                        melodic.append(thisChord.lyrics[2].text)
                        harmonic.append(thisChord.lyrics[3].text)
                    else:
                        print('?')
                elif(len(thisChord.lyrics) == 2):
                    print(thisChord.lyrics[0].text)
                    print(thisChord.lyrics[1].text)
                    melodic.append(thisChord.lyrics[0].text)
                    harmonic.append(thisChord.lyrics[1].text)
                elif len(thisChord.lyrics)== 0:  # empty, probably the problem of the software
                    print('?')
                else:
                    print('??')
                    #melodic.append(thisChord.lyrics[2].text)
                    #harmonic.append(thisChord.lyrics[3].text)
                #elif(thisChord.lyrics.__len__ == 2)
            melodic[0] = original[0]
            harmonic[0] = original[0]
            melodic = translate_annotation(original, melodic)
            harmonic = translate_annotation(original, harmonic)
            fmelodic = open(cwd_annotation_m + 'translated_' + fn[10:13] + 'melodic.txt', 'w')
            fharmonic = open(cwd_annotation_h + 'translated_' + fn[10:13] + 'harmonic.txt', 'w')
            if len(melodic) != len(harmonic):
                input('melodic and harmonic different lengths?!')
            else:
                for i, item in enumerate(melodic):
                    print(melodic[i].encode('utf-8').decode('ansi'), end=' ', file=fmelodic)
                    print(harmonic[i].encode('utf-8').decode('ansi'), end=' ', file=fharmonic)
def translate_annotation(ori, cur):
    for i, item in enumerate(cur):  # translate melodic, harmonic into chord labels:
        if cur[i] == '.' or cur[i] == '_':
            cur[i] = ori[i]
        if i < len(cur) - 1:
            if(cur[i + 1] == '<'):
                cur[i + 1] = cur[i]
        if(cur[i] == '>'):
            if cur[i + 1] == '.' or cur[i + 1] == '_':
                cur[i] = ori[i + 1]
            else:
                cur[i] = cur[i + 1]
    for i, item in enumerate(cur):
        if(item in '._<>,' or item.find(',') != -1 or item.find('.') != -1 or item.find('_') != -1 or item.find('<') != -1 or item.find('>') != -1 or item.find('?') != -1):
            print(i, item, cur)
            input('?')

    return cur
    """
    
    translate < > . _ into chord labels
    :return:
    """
def write_to_files(transposed='', multi=0):
    """
    A function that can either translate transposed files or not.
    :param transposed:
    :param multi: a switch that whether generate multiple interpretation or not
    :return: 
    """
    num_of_ambiguity = 0
    bad = 0
    num_of_samples = 0
    for fn in os.listdir(cwd):
        if fn[-3:] == 'mid':
            print(fn)
            if ((os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[
                                                                                                    0:3] + '.pop''')) or
                    (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[
                                                                                                        0:3] + '.pop.not'''))):

                s = converter.parse(cwd + fn)
                sChords = s.chordify()
                print(len(sChords.notes))
                num_of_samples += len(sChords.notes)
                beat_poly = ['0'] * len(sChords.notes)
                num_of_salami_poly = 0
                num_of_salami_chord = 0
                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    beat_poly[i] = thisChord.beatStr  # the ptr is not accurate, but the off-beat can still be detected
                    if (len(thisChord.beatStr) != 1):
                        print('beat location is: ' + thisChord.beatStr)

            if (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop''')):
                f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop', 'r')
                file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop'
                if(multi==0):
                    fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + transposed + fn[
                                                                                                                0:3] + '.pop',
                            'w')
                else:
                    fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_multi' + transposed + fn[
                                                                                                                 0:3] + '.pop',
                                'w')
            elif (
            os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop.not''')):
                f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop.not', 'r')
                file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + transposed + fn[0:3] + '.pop.not'
                if(multi==0):
                    fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + transposed + fn[
                                                                                                                0:3] + '.pop.not',
                            'w')
                else:
                    fnew = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_multi' + transposed + fn[
                                                                                                                 0:3] + '.pop.not',
                                'w')
            else:
                print('no file is opened')
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line, num_of_salami_poly, num_of_salami_chord, bad, num_of_ambiguity = translate_chord_line(num_of_salami_poly,
                                                                                          num_of_salami_chord, line,
                                                                                          replace, beat_poly, bad, fn, num_of_ambiguity, multi)
                print(line)
                if ('[' in line) or (']' in line) or ('(' in line) or (')' in line):
                    print(fn)
                    print(line)
                    print('what do you think?')
                    # input('[]() still exists, bug!')  # exmaine
                    line = re.sub(r'\[\w\]', '', line)
                    line = re.sub(r'\[\w\w]', '', line)
                    line = re.sub(r'\[\w\W]', '', line)

                    line = re.sub(r'\[\w \w]', '', line)  # nesty exceptions
                    print('new line:' + line)
                    # input('what do you think?')
                    # pattern = re.compile(r'\w[+]')
                    # if(pattern):  # deal with  'c[d]' case
                print(line, end='\n', file=fnew)
    print('bad =' + str(bad))
    print('number of samples = ' + str(num_of_samples))
    print('number of ambiguity = ' + str(num_of_ambiguity))
if __name__ == "__main__":
    #multi = int(input("Do you want multiple interpretations or not? (1 yes 0 no)"))
    #write_to_files(multi=multi)
    annotation_to_txt()
    #write_to_files('transposed_')
