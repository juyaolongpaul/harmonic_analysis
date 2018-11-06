from music21 import *
import os
import re
import numpy as np
dic = {}
from counter_chord_frequency import *
from adding_window_one_hot import adding_window_one_hot
from test_musicxml_gt import get_chord_tone
from random import shuffle
format = ['mid']
cwd = '.\\bach_chorales_scores\\transposed_MIDI\\'
x = []
y = []


def get_chord_line(line, sign):
    """

    :param line:
    :param replace:
    :return:
    """
    for letter in '!':
        line = line.replace(letter, '')
    #if(sign == '0'):  # now, no inversions what so ever
    #line = re.sub(r'/\w[b#]*', '', line)  # remove inversions + shapr + flat
    # careful that / can either represent inversions as well as half diminished sign!
    return line


def calculate_freq(dic, line):
    """
    :param dic:
    :param line:
    :return:
    """
    for chord in line.split():
        dic.setdefault(chord, 0)
        dic[chord] += 1
    return dic


def output_freq_to_file(filename, dic):
    """

    :param filename:
    :param dic:
    :return:
    """
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    fchord = open(filename, 'w')
    total_freq = 0
    total_percentage = 0
    for word in li:
        total_freq += word[1]
    for word in li:
        print(word, end='', file=fchord)
        total_percentage += word[1] / total_freq
        print(str(word[1] / total_freq), end='', file=fchord)
        print(' total: ' + str(total_percentage), file=fchord)


def fill_in_chord_class(chord, chordclass, list):
    """

    :param chordclass: The chord class vector that needs to label
    :param list: The chord list with outputdim top freq
    :return: the modified pitch class that need to store
    """
    for i, chord2 in enumerate(list):
        if(chord == chord2):

            chordclass[i] = 1
            break
    empty = [0] * (len(list)+1)
    if(chordclass == empty):  # this is 'other' chord!
        chordclass[len(list)] = 1
    return chordclass


def y_non_chord_tone(chord, chordclass, list):
    """

    :param chordclass: The chord class vector that needs to label
    :param list: The chord list with outputdim top freq
    :return: the modified pitch class that need to store
    """
    for i, chord2 in enumerate(list):
        if(chord == chord2):

            chordclass[i] = 1
            break
    empty = [0] * (len(list)+1)
    if(chordclass == empty):  # this is 'other' chord!
        chordclass[len(list)] = 1
    return chordclass


def fill_in_pitch_class_with_bass(pitchclass, list, counter):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """

    pitchclassvoice = pitchclass
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    for i in list:
        pitchclass[i] = 1
    pitchclassvoice[0] = pitchclass
    if len(list) == 4:
        pitchclassvoice[1][list[3]] = 1  # the last voice is the bass, proved
    else:
        print('no bass?')
        counter += 1
        print(counter)
    pitchclassvoice = pitchclassvoice.ravel()
    pitchclassvoice = pitchclassvoice.tolist()
    return pitchclassvoice, counter


def fill_in_pitch_class_with_octave(list):
    """
    Put pitch in a compressed range
    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    LOWEST_PITCH_CONSIDERED = 36
    PITCH_RANGE = 48
    SEMITONE_IN_OCTAVE = 12
    pitchclass = [0] * PITCH_RANGE # calculate by pitch_distribution
    for i in list:
        midi = i.midi
        while midi-LOWEST_PITCH_CONSIDERED >= len(pitchclass):
            midi -= SEMITONE_IN_OCTAVE # move an octave lower until it falls into the range
        while midi < LOWEST_PITCH_CONSIDERED:
            midi += SEMITONE_IN_OCTAVE # move an octave higher until it falls into the range
        pitchclass[midi-LOWEST_PITCH_CONSIDERED] = 1  # the lowest is 28, compressed
    return pitchclass


def pitch_distribution(list, counter, countermin):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    for i in list:
        if i.midi > counter:
            counter = i.midi
            print('max', counter)
        if i.midi < countermin:
            countermin = i.midi
            print('min', countermin)
        #if(i.midi > 84):
            #input('pitch class more than 84')
    return counter, countermin


def fill_in_pitch_class_with_voice(pitchclass, list):
    """

    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number
    :return: the modified pitch class that need to store
    """
    pitchclassvoice = pitchclass
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))
    pitchclassvoice = np.vstack((pitchclassvoice, pitchclass))  # 2-D array

    for i, item in enumerate(list):
        pitchclassvoice[i][item] = 1
    pitchclassvoice = pitchclassvoice.ravel()
    pitchclassvoice = pitchclassvoice.tolist()
    return pitchclassvoice


def get_pitch_class_for_four_voice(thisChord, s):
    pitch_class_four_voice = []

    for j, part in enumerate(s.parts):
        for i in range(len(part.measure(
                thisChord.measureNumber).notes)):  # didn't work correctly if using enumerate, internal bug!!!!!
            # print("voice:", j, "in measure", thisChord.measureNumber, ',', i, "note in this voice", "pitch is",
            # part.measure(thisChord.measureNumber).notes[i].pitch,
            # len(part.measure(thisChord.measureNumber).notes), "notes in total", "note.beatStr:", part.measure(thisChord.measureNumber).notes[i].beatStr,
            # "thisChord.beatStr", thisChord.beatStr)
            if part.measure(thisChord.measureNumber).notes[i].beatStr == thisChord.beatStr:

                pitch_class_four_voice.append(part.measure(thisChord.measureNumber).notes[i].pitch.pitchClass)
            else:
                if part.measure(thisChord.measureNumber).notes[i].beatStr < thisChord.beatStr:
                    if len(part.measure(thisChord.measureNumber).notes) == i + 1:
                        pitch_class_four_voice.append(part.measure(thisChord.measureNumber).notes[i].pitch.pitchClass)
                    elif i + 1 < len(part.measure(thisChord.measureNumber).notes):
                        if part.measure(thisChord.measureNumber).notes[i + 1].beatStr > thisChord.beatStr:
                            pitch_class_four_voice.append(part.measure(thisChord.measureNumber).notes[i].pitch.pitchClass)
    return pitch_class_four_voice


def fill_in_pitch_class_4_voices(pitchclass, list, thisChord, s):
    """
    Generate one-hot encoding for 4 voices using pitch classes
    :param pitchclass:
    :param list:
    :param thisChord:
    :param s:
    :return:
    """
    pitchclass = [0] * 48
    if len(list) <4:
        list_ori = list
        print('originally unique pitch is less than 4', list)
        list = get_pitch_class_for_four_voice(thisChord, s) # in case there are only less then 4 voices
        print('after processing', list)
        for item in list: # TODO: solve this hacky fix later! you function above do not work perfectly!
            if item not in list_ori:
                list.remove(item)
        if len(list) > 4:
            list = list[:(4-len(list))]
    for i, item in enumerate(list):
        pitchclass[i*12 + item] = 1
    return pitchclass


def fill_in_pitch_class_binary(pitchclass, list, thisChord, s, bad):
    """
    Encode pitch-class information for each voice in a binary encoding
    :param pitchclass: Pitch class binary encodings for each voice
    :param list:
    :return:
    """
    if len(list) <4:
        list_ori = list
        print('originally unique pitch is less than 4', list)
        list = get_pitch_class_for_four_voice(thisChord, s) # in case there are only less then 4 voices
        print('after processing', list)
        for item in list: # TODO: solve this hacky fix later! you function above do not work perfectly!
            if item not in list_ori:
                list.remove(item)
                bad += 1
        if len(list) > 4:
            list = list[:(4-len(list))]
            bad += 1

    #print('list length:', len(list), 'content of the list:', list)
    for i, item in enumerate(list):
        binary_encoding = '{0:04b}'.format(item)
        for j, item2 in enumerate(binary_encoding): # each bin goes to the pitchclass vector
            #print(j, '+', 4*i, 'binary encoding:', binary_encoding)
            pitchclass[j + 4*i] = int(item2)
    return pitchclass, bad


def fill_in_pitch_class(pitchclass, list):
    """
    :param pitchclass: The pitch class vector that needs to label
    :param list: The pitch class encoded in number, [0,3,6,9]
    :return: the modified pitch class that need to store
    """
    for i in list:
        pitchclass[i] = 1
    return pitchclass


def fill_in_pitch_class_7(list, name):
    """
    Ignore accidentals
    :param name: pitch names
    :param list: The pitch class vector in 12 dim
    :return: the modified pitch class that need to store
    """
    NUM_OF_PITCH_CLASS = 12
    NUM_OF_GENERIC_PITCH_CLASS = 7
    pitchclass = [0] * int(NUM_OF_GENERIC_PITCH_CLASS * len(list)/NUM_OF_PITCH_CLASS)
    for i,item in enumerate(list):
        octave_or_voice = int(i/NUM_OF_PITCH_CLASS)  # find out it is which pitch or voice
        if item == 1:
            if (i%NUM_OF_PITCH_CLASS == 0):
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+0] = 1
            elif i%NUM_OF_PITCH_CLASS == 1:
                if 'C#' in name: # if the spell is C
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 0] = 1
                elif 'D-' in name:
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 1] = 1
                else:
                    input('no correct pitch spelling for 1?')
            elif i%NUM_OF_PITCH_CLASS == 2:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+1] = 1
            elif i%NUM_OF_PITCH_CLASS == 3:
                if 'D#' in name: # if the spell is C
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 1] = 1
                elif 'E-' in name:
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 2] = 1
                else:
                    input('no correct pitch spelling for 3?')  #  one exception: might be d## or F-
            elif i%NUM_OF_PITCH_CLASS == 4:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+2] = 1
            elif i%NUM_OF_PITCH_CLASS == 5:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+3] = 1
            elif i%NUM_OF_PITCH_CLASS == 6:
                if 'F#' in name: # if the spell is C
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 3] = 1
                elif 'G-' in name:
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 4] = 1
                else:
                    input('no correct pitch spelling for 6?')
            elif i%NUM_OF_PITCH_CLASS == 7:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+4] = 1
            elif i%NUM_OF_PITCH_CLASS == 8:
                if 'G#' in name: # if the spell is C
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 4] = 1
                elif 'A-' in name:
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 5] = 1
                else:
                    input('no correct pitch spelling for 8?')
            elif i%NUM_OF_PITCH_CLASS  == 9:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+5] = 1
            elif i%NUM_OF_PITCH_CLASS == 10:
                if 'A#' in name: # if the spell is C
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 5] = 1
                elif 'B-' in name:
                    pitchclass[octave_or_voice * NUM_OF_GENERIC_PITCH_CLASS + 6] = 1
                else:
                    input('no correct pitch spelling for 10?')
            elif i%NUM_OF_PITCH_CLASS == 11:
                pitchclass[octave_or_voice*NUM_OF_GENERIC_PITCH_CLASS+6] = 1
            else:
                input('pitch class cannot compress into 7?')
    return pitchclass


def get_chord_list(output_dim, sign):
    dic = {}
    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if file_name.find('translated_transposed') != -1:
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line = get_chord_line(line, sign)
                #print(line)
                dic = calculate_freq(dic, line)
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    list_of_chords = []
    for i, word in enumerate(li):
        if(i == output_dim - 1):  # the last one is 'others'
            break
        list_of_chords.append(word[0])
    print (list_of_chords)  # Get the top 35 chord freq
    return list_of_chords


def add_beat_into_binary(pitchclass, beat):
    if (len(beat) == 1):  # on beat
        pitchclass.append(1)
    else:
        pitchclass.append(0)
    return pitchclass


def add_beat_into(pitchclass, beat, inputtype):
    """
    adding two dimension to the input vector, specifying whether the current slice is on/off beat.
    :return:
    """
    if inputtype.find('2meter') != -1:
        if(len(beat) == 1):  # on beat
            pitchclass.append(1)
            pitchclass.append(0)
        else: # off beat
            pitchclass.append(0)
            pitchclass.append(1)
    elif inputtype.find('3meter') != -1:
        if (len(beat) == 1):  # on beat
            if beat == '1': # on a strong beat
                pitchclass.append(1)
                pitchclass.append(0)
                pitchclass.append(0)
            else: # on a weak beat
                pitchclass.append(0)
                pitchclass.append(1)
                pitchclass.append(0)
        else: # off beat
            pitchclass.append(0)
            pitchclass.append(0)
            pitchclass.append(1)
    return pitchclass


def get_non_chord_tone(x,y,outputdim):
    """
    Take out chord tones, only leave with non-chord tone

    :param x:
    :param y:
    :return:
    """
    yy = [0] * len(y)
    for i in range(outputdim):
        if(x[i] == 1 and y[i] == 0):  # non-chord tone
            if(y[-1] != 1):
                yy[i] = 1
        if(y[-1] == 1):  # occasion where the chord cannot be recognized
            if(yy == [0] * len(y)):
                yy[-1] = 1
            else:
                print('x:' , x)
                print('y:' , y)
                print('yy:' , yy)
                input('error')
            break
    return yy


def get_non_chord_tone_4_binary(x,y,outputdim, f):
    """

    :param x:
    :param y:
    :param outputdim:
    :param f:
    :return:
    """
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    yy = [0] * 4  # SATB
    yori = y
    y = y[:12]
    if yori[-1] == 1:  # broken chord, assume there is no non-chord tone
        print('n/a', end=' ', file=f)
        return yy

    chordtone_ID =[]
    for j, item in enumerate(y): # translate one hot into pitch ID
        if item == 1:
            chordtone_ID.append(j)
    for i in range(4):  # get the binary encoding for each voice and translate into pitch ID
        pitch_binary = x[4*i : 4*i + 4]
        pitch_binary_list = pitch_binary.tolist()
        pitch_str = str(int(pitch_binary_list[0])) + str(int(pitch_binary_list[1])) + str(int(pitch_binary_list[2])) + str(int(pitch_binary_list[3]))
        pitch_ID = int(pitch_str, 2)
        if pitch_ID in chordtone_ID: # current voice does not have non-chord tone
            yy[i] = 0
        else:
            yy[i] = 1 # current voice has a non-chord tone
    if yy == [0] * 4: # no non-chord tone
        print('n/a', end=' ', file=f)
        return yy
    else:  # there is non-chord tone
        for i, item in enumerate(yy) :  # examine yy
            if item == 1: # current voice is a non-chord tone
                pitch_binary = x[4 * i: 4 * i + 4]
                pitch_binary_list = pitch_binary.tolist()
                pitch_str = str(int(pitch_binary_list[0])) + str(int(pitch_binary_list[1])) + str(
                    int(pitch_binary_list[2])) + str(int(pitch_binary_list[3]))
                pitch_ID = int(pitch_str, 2)
                print(pitchclass[pitch_ID], end='', file=f)
        print(end=' ', file=f) # we want dfg format as non-chord tone format
        return yy


def get_non_chord_tone_4(x,y,outputdim, f):
    """
    Take out chord tones, only leave with non-chord tone

    :param x:
    :param y:
    :return:
    """
    pitchclass = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    nonchordpitchclassptr = [-1] * 4
    yori = y
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    for i in range(len(x) - 2):
        if(yori[-1] == 1): # broken chord, assume there is no non-chord tone!
            break
        if(x[i] == 1):  # go through the present pitch class
            yyptr += 1
            if(y[i%12] == 0):  # the present pitch class is a chord tone or not
                #print('yyptr:' , yyptr)
                #if(yyptr == 4):
                    #print('debug')
                yy[yyptr] = 1
                nonchordpitchclassptr[yyptr] = i%12
    if(nonchordpitchclassptr == [-1] * 4):
        print('n/a', end= ' ', file=f)
    else:
        #if(2 in nonchordpitchclassptr):
            #print('debug')
        for item in nonchordpitchclassptr:
            if(item != -1):
                print(pitchclass[item], end='', file=f) # we want dfg, not d f g!
        print(end=' ', file=f)
    return yy


def pitch_class_7_to_12(ori):
    """
    Return the id of pitch class from generic pitch class
    :param ori:
    :return:
    """
    if ori == 0:
        return [0,1]
    elif ori == 1:
        return [2,3]
    elif ori == 2:
        return [4]
    elif ori == 3:
        return [5,6]
    elif ori == 4:
        return [7,8]
    elif ori == 5:
        return [9,10]
    elif ori == 6:
        return [11]


def get_non_chord_tone_4_pitch_class_7(x,y,outputdim, f):
    """
    Take out chord tones, only leave with non-chord tone

    :param x: it is the X
    :param y: it is all the chord tones in pitch-class ID
    :return:
    """
    pitchclass = ['c','d','e','f','g','a','b']
    nonchordpitchclassptr = [-1] * 4
    yori = y
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    for i in range(len(x) - 2):
        if(yori[-1] == 1): # broken chord, assume there is no non-chord tone!
            break
        if(x[i] == 1):  # go through the present pitch class
            yyptr += 1
            real_i = pitch_class_7_to_12(i)
            if len(real_i) == 1:
                if(y[real_i[0]%12] == 0):  # the present pitch class is a chord tone or not
                    #print('yyptr:' , yyptr)
                    #if(yyptr == 4):
                        #print('debug')
                    yy[yyptr] = 1
                    nonchordpitchclassptr[yyptr] = i%12
            elif len(real_i) == 2:
                if (y[real_i[0] % 12] == 0 and y[real_i[1] % 12] == 0):  # the present pitch class is a chord tone or not
                    # print('yyptr:' , yyptr)
                    # if(yyptr == 4):
                    # print('debug')
                    yy[yyptr] = 1
                    nonchordpitchclassptr[yyptr] = i % 12
    if(nonchordpitchclassptr == [-1] * 4):
        print('n/a', end= ' ', file=f)
    else:
        #if(2 in nonchordpitchclassptr):
            #print('debug')
        for item in nonchordpitchclassptr:
            if(item != -1):
                print(pitchclass[item], end='', file=f) # we want dfg, not d f g!
        print(end=' ', file=f)
    return yy


def get_non_chord_tone_4_music21(x, y, f, thisChord):
    '''
    Getting crappy but consistent non-chord tone labels from music21 module. Four triads and five types of 7th chords
    are considered. Otherwise they are all non-chord tones.
    :param x:
    :param y:
    :return:
    '''
    allowed_qualities = [[0, 4, 7], [0, 3, 6], [0, 3, 7], [0, 4, 8], [0, 3, 6, 9], [0, 3, 6, 10], [0, 3, 7, 10],
                         [0, 4, 7, 10], [0, 4, 7, 11]]
    pitchclass = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    nonchordpitchclassptr = [-1] * 4
    yori = y
    y = y[:12]
    yy = [0] * 4
    yyptr = -1
    if (thisChord.normalForm in allowed_qualities or thisChord.pitchedCommonName.find(
            'incomplete') != -1
            or ((thisChord.pitchedCommonName.find(
                'dominant') != -1 or thisChord.pitchedCommonName.find(
                'diminished') != -1 or thisChord.pitchedCommonName.find('major') != -1
                 or thisChord.pitchedCommonName.find(
                        'half-diminished') != -1 or thisChord.pitchedCommonName.find('minor') != -1)
                and thisChord.pitchedCommonName.find('seventh') != -1)):
        print('this slice, music21 consider no nct!')
    else:  # they are all non-chord tones
        for i in range(len(x) - 2):
            if (yori[-1] == 1):  # broken chord, assume there is no non-chord tone!
                break
            if (x[i] == 1):  # go through the present pitch class
                yyptr += 1
                #if (y[i % 12] == 0):  # the present pitch class is a chord tone or not
                    # print('yyptr:' , yyptr)
                #if (yyptr == 4):
                    #print('debug')
                yy[yyptr] = 1
                nonchordpitchclassptr[yyptr] = i % 12
    if (nonchordpitchclassptr == [-1] * 4):
        print('n/a', end=' ', file=f)
    else:
        #if (2 in nonchordpitchclassptr):
            #print('debug')
        for item in nonchordpitchclassptr:
            if (item != -1):
                print(pitchclass[item], end='', file=f)
        print(end=' ', file=f)
    return yy


def determine_middle_name(augmentation, source, portion, pitch):
    '''
    Determine the file name of whether using augmentation, pitch or pitch-class and melodic or harmonic
    :param augmentation:
    :param pitch:
    :param source:
    :return:
    '''

    music21 = ''


    if (augmentation == 'Y'):
        if portion ==  'train':
            keys = '12keys'
        else:
            keys = 'keyOri'
    elif pitch.find('oriKey') == -1:
        keys = 'keyC'
    else:
        keys = 'keyOri'
    return keys, music21

def determine_middle_name2(augmentation, source, pitch):
    '''
    Only used for finding right np file for training ML model
    :param augmentation:
    :param pitch:
    :param source:
    :return:
    '''

    music21 = ''


    if (augmentation == 'Y'):
            keys = '12keys'
            keys1 = 'keyOri'
    elif pitch.find('oriKey') == -1:
        keys = 'keyC'
        keys1 = 'keyC'
    else:
        keys = keys1 = 'keyOri'
    return keys, keys1, music21

def find_id(input, version):
    """

    Find three digit of the labeled chorales in the folder
    :param input:
    :return:
    """
    import re
    rameau_crap = ['131', '142', '146', '150',
                   '161', '253', '357', '359', '361',
                   '362', '363', '365', '366', '367', '368', '369', '370', '371']
    id_sum = []
    p = re.compile(r'\d{3}')
    for fn in os.listdir(input):
        if fn.find('translated') == -1:  # only look for non-"chor" like annotation files
            continue
        id = p.findall(fn)
        if(id != []):
            id_sum.append(id[0])
            #print(id[0])
    id_sum_strip = []
    [id_sum_strip.append(i) for i in id_sum if not i in id_sum_strip]
    id_sum_strip.remove('316')  # this file does not align
    #id_sum_strip = ['001','002','003','004','005','006','007','008','010','012',]
    # delete all these crap files
    if version == 153:
        if '130' in id_sum_strip:
            id_sum_strip.remove('130')
        if '130' in id_sum_strip:
            id_sum_strip.remove(
                '133')  # remove these bad files where the input and output do not match (version problem)
        for i in rameau_crap:
            if i in id_sum_strip:
                id_sum_strip.remove(i)
    return id_sum_strip


def generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input1, f1, output, f2, sign, predict, augmentation, pitch, data_id, times, portion, outputtype, data_id_total, inputtype):
    """
    Generate non-chord tone verserion of the annotations and put them into matrice for machine learning
    as long as the file ID is given
    :param counter1:
    :param counter2:
    :param x:
    :param y:
    :param inputdim:
    :param outputdim:
    :param windowsize:
    :param counter:
    :param countermin:
    :param input:
    :param f1:
    :param output:
    :param f2:
    :param sign:
    :param predict:
    :param augmentation:
    :param pitch:
    :param data_id:
    :return:
    """
    fn_total_all = [] # this save all file ID, including training, validation and test data
    fn_total = [] # this only includes one of the following three: training, validation and test data
    file_counter = 0
    slice_counter = 0
    keys, music21 = determine_middle_name(augmentation, sign, portion, pitch)
    number = len(data_id)
    if sign == 'Rameau':
        input1 = os.path.join('.', 'bach_chorales_scores', 'original_midi+PDF')
        f1 = '.mid'
    if sign == 'rule_MaxMel':
        label = 'chor'
    else:
        label = 'Chorales_Bach_'
    if not os.path.isdir(os.path.join('.', 'data_for_ML', sign)):
        os.mkdir(os.path.join('.', 'data_for_ML', sign))
    search_file_x = os.path.join('.', 'data_for_ML', sign, sign) + '_x_windowing_' + str(windowsize) + outputtype+ pitch + inputtype + '_New_annotation_' + keys +'_' +music21+'_' + portion + 'ing' + str(number) + '_cv_' + str(times) + '.txt'
    search_file_y = os.path.join('.', 'data_for_ML', sign, sign) + '_y_windowing_' + str(
        windowsize) + outputtype + pitch + inputtype + '_New_annotation_' + keys + '_' + music21 + '_' + portion + 'ing' + str(
        number) + '_cv_' + str(times) + '.txt'
    if not (os.path.isfile(search_file_x)): # if there matrix file is already there, no need to generate again. Although each shuffle, the content will be different
        for id, fn in enumerate(os.listdir(input1)):
                if fn.find('KB') != -1 and fn[-4:] == f1:
                    p = re.compile(r'\d{3}')  # find 3 digit in the file name
                    id_id = p.findall(fn)
                    if id_id[0] in data_id:  # if the digit found in the list, add this file
                        if(augmentation != 'Y'):  # Don't want data augmentation in 12 keys
                            if pitch.find('oriKey') == -1: # we want transposed key
                                if(fn.find('cKE') != -1 or fn.find('c_oriKE') != -1):  # only wants key c
                                    fn_total.append(fn)
                            else:
                                if fn.find('_ori') != -1:  # no transposition
                                    fn_total.append(fn)
                        elif augmentation == 'Y' and portion == 'train':
                            fn_total.append(fn)  # we want 12 keys on training set
                        elif augmentation == 'Y' and (portion == 'valid' or portion == 'test'): # original keys on the valid and test set:
                            if (fn.find('_ori') != -1): # only add original key
                                fn_total.append(fn)
                    if id_id[0] in data_id_total:
                        # This section of code aims to add all the file IDs across training validation and test set
                        if(augmentation != 'Y'):  # Don't want data augmentation in 12 keys
                            if pitch.find('oriKey') == -1: # we want transposed key
                                if(fn.find('cKE') != -1 or fn.find('c_oriKE') != -1):  # only wants key c
                                    fn_total_all.append(fn)
                            else:
                                if fn.find('_ori') != -1:  # no transposition
                                    fn_total_all.append(fn)
                        elif augmentation == 'Y':
                            fn_total_all.append(fn)  # we want 12 keys on all sets

        if(predict == 'N'):
            shuffle(fn_total)  # shuffle (by chorale) on the training and validation set
        print (fn_total)
        #input('?')
        bad_voice_finding_slice = 0
        # The following part calculates chord frequency distribution
        dic = {}  # Save chord name + frequencies
        for id, fn in enumerate(fn_total_all):
            ptr = p.search(fn).span()[0]  # return the starting place of "001"
            ptr2 = p.search(fn).span()[1]
            if (os.path.isfile(os.path.join(output, fn[:ptr]) + 'translated_' + label + fn[ptr:ptr2] + sign + f2)):
                f = open(os.path.join(output, fn[:ptr]) + 'translated_' + label + fn[ptr:ptr2] + sign + f2, 'r')
            else:
                continue  # skip the file which does not have chord labels
            for line in f.readlines():
                line = get_chord_line(line, sign)
                dic = calculate_freq(dic, line)
        li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        list_of_chords = []
        for i, word in enumerate(li):
            list_of_chords.append(word[0]) # Get all the chords
        f_chord = open('chord_freq.txt', 'w')
        for item in li:
            print(item, file=f_chord)
        f_chord.close()
        f_chord2 = open('chord_name.txt', 'w')
        for item in list_of_chords:
            print(item, file=f_chord2)  # write these chords into files, so that we can have chords name for
            # confusion matrix
        f_chord2.close()
        for id, fn in enumerate(fn_total):
                print(fn)
                ptr = p.search(fn).span()[0]  # return the starting place of "001"
                ptr2 = p.search(fn).span()[1]
                chorale_x = []
                chorale_x_12 = []  # This is created to store 12 pitch class encoding when generic (7)
                # pitch class is used. This one is used to indicate which one is NCT.
                if (os.path.isfile(
                        os.path.join(output, fn[:ptr]) + 'translated_' + label + fn[ptr:ptr2] + sign + f2)):
                    f = open(
                        os.path.join(output, fn[:ptr]) + 'translated_' + label + fn[ptr:ptr2] + sign + f2,
                        'r')
                    f_non = open(os.path.join(output, fn[:ptr]) + label + 'non_chord_tone_' + music21 + '_' + sign + pitch
                                 + fn[ptr:ptr2] + f2, 'w')
                else:
                    continue  # skip the file which does not have chord labels
                file_counter += 1
                s = converter.parse(os.path.join(input1, fn))
                sChords = s.chordify()
                slice_input = 0
                #print(slice_input)
                #length = len(sChords)
                thisChordAll = []
                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    thisChordAll.append(thisChord)
                    slice_input += 1
                    counter1 += 1
                    slice_counter += 1
                    if pitch != 'pitch_class_binary':
                        pitchClass = [0] * inputdim
                        #pitchClass, counter = fill_in_pitch_class_with_bass(pitchClass, thisChord.pitchClasses, counter)
                        if(pitch == 'pitch' or pitch == 'pitch_7'):
                            pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)
                        elif pitch == 'pitch_class' or pitch == 'pitch_class_7':
                            pitchClass= fill_in_pitch_class(pitchClass, thisChord.pitchClasses)
                        elif pitch == 'pitch_class_4_voices' or pitch == 'pitch_class_4_voices_7':
                            pitchClass = fill_in_pitch_class_4_voices(pitchClass, thisChord.pitchClasses, thisChord, s)
                        if pitch.find('pitch') != -1 and pitch.find('7') != -1: # Use generic pitch, could be
                            # just pitch, pitch_class or pitch_class in 4 voices. Append 7 in the end
                            pitchClass_12 = pitchClass  # pitchClass saves the original
                            pitchClass = fill_in_pitch_class_7(pitchClass, thisChord.pitchNames)
                        pc_counter = 0
                        for ii in pitchClass:
                            if ii == 1:
                                pc_counter +=1
                        if (pc_counter > 4):
                            print("pc is greate than 4!~")
                        counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                        #pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)  # add voice leading (or not)
                        #(thisChord.pitchClasses)
                        pitchClass = add_beat_into(pitchClass, thisChord.beatStr, inputtype)  # add on/off beat info
                        if pitch.find('pitch') != -1 and pitch.find('7') != -1:  # add beat info for 12 pitch class
                            # if generic pitch is used
                            pitchClass_12 = add_beat_into(pitchClass_12, thisChord.beatStr, inputtype)
                    else:  # if binary encoding is used, each voice is specified with a pitch-class
                        input('binary encoding is depreciated!')
                        # if inputdim > 8 and inputdim <= 16:
                        #     pitchClass = [0] * 16
                        # else:
                        #     input('input_dim is not within [8,16]!')
                        #
                        # pitchClass, bad_voice_finding_slice = fill_in_pitch_class_binary(pitchClass, thisChord.pitchClasses, thisChord, s, bad_voice_finding_slice)  # pitchClass is sorted in SATB, respectively
                        # counter, countermin = pitch_distribution(thisChord.pitches, counter, countermin)
                        # # pitchClass = fill_in_pitch_class_with_octave(thisChord.pitches)  # add voice leading (or not)
                        # # (thisChord.pitchClasses)
                        # pitchClass = add_beat_into_binary(pitchClass, thisChord.beatStr)  # add on/off beat info
                    if(i == 0):
                        if pitch.find('pitch') != -1 and pitch.find('7') != -1:
                            chorale_x_12 = np.concatenate((chorale_x_12, pitchClass_12))
                        chorale_x = np.concatenate((chorale_x, pitchClass))
                    else:
                        if pitch.find('pitch') != -1 and pitch.find('7') != -1:
                            chorale_x_12 = np.vstack((chorale_x_12, pitchClass_12))
                        chorale_x = np.vstack((chorale_x, pitchClass))
                chorale_x_window = adding_window_one_hot(chorale_x, windowsize)
                if(file_counter == 1):
                    x = chorale_x_window
                else:
                    x = np.concatenate((x, chorale_x_window))
                slice_counter = 0  # remember what slice in order to get the pitch class info
                for line in f.readlines():
                    line = get_chord_line(line, sign)
                    for chord in line.split():
                        if(chord.find('g]') != -1):
                            print(fn)
                            input1('wtf is that?')
                        counter2 += 1
                        #chord_class = [0] * outputdim
                        #chord_class = y_non_chord_tone(chord, chord_class, list_of_chords)
                        #chord_class = get_non_chord_tone(chorale_x[slice_counter],)
                        if outputtype == "NCT":
                            chord_class = get_chord_tone(chord, outputdim)
                            #chord_class = get_non_chord_tone(chorale_x[slice_counter], chord_class, output_dim)
                            if pitch != 'pitch_class_binary':
                                if pitch.find('pitch') != -1 and pitch.find('7') != -1:
                                    chord_class = get_non_chord_tone_4(chorale_x_12[slice_counter], chord_class, outputdim,
                                                                       f_non)  # Here we assume NCT result is the same
                                    # no matter whether 12 pitch class or 7 is used
                                else:
                                    chord_class = get_non_chord_tone_4(chorale_x[slice_counter], chord_class, outputdim, f_non)

                            else:
                                chord_class = get_non_chord_tone_4_binary(chorale_x[slice_counter], chord_class, outputdim, f_non)
                        elif outputtype == 'CL':
                            chord_class = [0] * len(list_of_chords)
                            chord_class = fill_in_chord_class(chord, chord_class, list_of_chords)
                        slice_counter += 1
                        if(counter2 == 1):
                            y = np.concatenate((y, chord_class))
                        else:
                            y = np.vstack((y, chord_class))
                print('slices of output: ', slice_counter, "slices of input", slice_input)
                if abs(slice_counter - slice_input) >= 1 and slice_counter != 0:
                    input('fix this or delete this')
                # save x, y into file as "ground truth"

        # add zero to the matrix, so that it can be divided by 50
        print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
        print("original x, y: " + str(x.shape[0]) + str(y.shape[0]))
        yy = [0] * outputdim
        yy[-1] = 1  # there should be a chord for these artificial slices, chord is 'other'
        print('yy:', yy)
        '''while(x.shape[0] % 50 !=0):
            x = np.vstack((x, [0] * inputdim))
            y = np.vstack((y, yy))'''
        print("Now x, y: " + str(x.shape[0]) + str(y.shape[0]))
        #np.savetxt('.//data_for_ML//' + sign + '_x_windowing_' + str(windowsize) + 'y4_non-chord_tone_pitch_class_New_annotation_12keys_150.txt', x, fmt = '%.1e')


        np.savetxt(search_file_x, x, fmt='%.1e')
        np.savetxt(search_file_y, y, fmt = '%.1e')
        print('number of bad slices', bad_voice_finding_slice)
def get_id(id_sum, num_of_chorale, times):
    """
    Get chorale ID for different batch of cross validation
    :param id_sum:
    :param num_of_chorale:
    :param times:
    :return:
    """
    placement = int(num_of_chorale / 10)
    placement2 = int(num_of_chorale / 10)
    valid_id = id_sum[times * placement2:(times + 1) * placement2]
    if (times != 9):
        test_id = id_sum[((times + 1)) * placement2:((times + 2)) * placement2]
    else:
        test_id = id_sum[((times + 1) % 10) * placement2:((times + 2) % 10) * placement2]
    if (times * placement != 0):
        if (times != 9):
            train_id = id_sum[:times * placement] + id_sum[(times + 2) * placement:]
        else:
            train_id = id_sum[((times + 2) % 10) * placement2:times * placement2]
    else:
        train_id = id_sum[((times + 2) % 10) * placement:]
    return train_id, valid_id, test_id


def generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1, output, f2, sign, augmentation, pitch, ratio, cv, version, distributed, outputtype, inputtype):
    """
    The only difference with "generate_data_windowing_non_chord_tone"
    :param counter1:
    :param counter2:
    :param string:
    :param string1:
    :param string2:
    :param x:
    :param y:
    :param inputdim:
    :param outputdim:
    :param windowsize:
    :param counter:
    :param countermin:
    :return:
    """
    id_sum = find_id(output, version)
    num_of_chorale = len(id_sum)
    #train_num = int(num_of_chorale * ratio)
    if distributed == 0: # generate CV matrices in a serialized way
        for times in range(cv):  # do cross validation to get file ID
            train_id, valid_id, test_id = get_id(id_sum, num_of_chorale, times)
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                                    output, f2, sign, 'N', augmentation, pitch, train_id, times+1, 'train', outputtype, id_sum, inputtype)  # generate training + validating data
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                          output, f2, sign, 'N', augmentation, pitch, valid_id, times+1, 'valid', outputtype, id_sum, inputtype)  # generate training + validating data
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                                    output, f2, sign, 'Y', augmentation, pitch, test_id, times+1, 'test', outputtype, id_sum, inputtype)  # generating test data
        #print('debug')
    else:
        for times in range(distributed-1, distributed):  # do cross validation to get file ID
            train_id, valid_id, test_id = get_id(id_sum, num_of_chorale, times)
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                                    output, f2, sign, 'N', augmentation, pitch, train_id, times+1, 'train', outputtype, id_sum, inputtype)  # generate training + validating data
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                          output, f2, sign, 'N', augmentation, pitch, valid_id, times+1, 'valid', outputtype, id_sum, inputtype)  # generate training + validating data
            generate_data(counter1, counter2, x, y, inputdim, outputdim, windowsize, counter, countermin, input, f1,
                                    output, f2, sign, 'Y', augmentation, pitch, test_id, times+1, 'test', outputtype, id_sum, inputtype)  # generating test data
if __name__ == "__main__":
    counter = 0
    counterMin = 60
    # Get input features
    sign = '0'#input("do you want inversions or not? 1: yes, 0: no")
    output_dim =  '12'#input('how many kinds of chords do you want to calculate?')
    window_size = '0'#int(input('how big window?'))
    '''sign = 0
    output_dim = 30
    window_size = 1'''
    output_dim = int(output_dim)
    input_dim = 12
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
        if(i == output_dim - 1):  # the last one is 'others'
            break
        list_of_chords.append(word[0])
    print (list_of_chords)  # Get the top 35 chord freq
    # Get the on/off beat info
    # Get the encodings for input
    counter1 = 0  # record the number of salami slices of poly
    counter2 = 0  # record the number of salami slices of chords
    input = '.\\bach-371-chorales-master-kern\\kern\\'
    output = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Melodic\\'
    f1 = '.xml'
    f2 = '.txt'
    generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, input_dim, output_dim, 2,
                                           counter, counterMin, input, f1, output, f2, 'melodic', 'Y')  # Y means predict the result for 6 chorales, no shuffle!





