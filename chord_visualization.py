from music21 import *
import os
import sys
format = ['krn']
cwd = '.\\bach-371-chorales-master-kern\\kern\\'
from get_input_and_output import get_chord_line
def put_non_chord_tone_into_musicXML(input, output, sign, f1, f2, pitch):
    """

    :param string:
    :param string1:
    :param string2:
    :return:
    """
    fn_total = []
    for id, fn in enumerate(os.listdir(input)):
        #print(fn)
        #if fn.find( 'KB') != -1 and fn[-4:] == f1 and fn.find('130') == -1 and fn.find('133') == -1 and fn.find('19') != -1:  # only look for the transposed one

        if fn.find('KBcKE') != -1 and fn[-4:] == f1:
            #if fn.find('340') != -1 or fn.find('358') != -1 or fn.find('362') != -1 or fn.find('003') != -1 or fn.find('008') != -1 or fn.find('014') != -1:
                fn_total.append(fn)
    for id, fn in enumerate(fn_total):
        ptr = fn.find('chor')
        if fn[-3:] == 'xml':
            if (os.path.isfile(output + fn[:ptr] + 'translated_' + fn[-7:-4] + '_'+ sign + f2) and os.path.isfile('.\\predicted_result\\' + 'predicted_result_' + fn + '_non-chord_tone_' + sign + '.txt')): # if the annotation file exists
                f = open(output + fn[:ptr] + 'non_chord_tone_' + '_' + sign + fn[-7:-4] + f2,'r')
                fchord = open(output + fn[:ptr] + 'translated_' + fn[-7:-4] + '_'+ sign + f2,'r')
                fprediction = open('.\\predicted_result\\' + 'predicted_result_' + fn + '_non-chord_tone_' + sign + pitch + '.txt', 'r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            print(fn[4:7])
            sChords = s.chordify()
            lineTotal = ''
            lineTotalNoInversion = ''
            linechordtotal = ''
            for linepre in fprediction.readlines():
                linepre = get_chord_line(linepre, sign)
                lineTotal += linepre
            chordpreTotal = lineTotal.split()
            lineTotal = ''
            for line in f.readlines():
                line = get_chord_line(line, sign)
                lineNoInversion = get_chord_line(line, '0')
                lineTotal += line
                lineTotalNoInversion += lineNoInversion
            chordTotal = lineTotal.split()
            chordTotalNoInversion = lineTotalNoInversion.split()
            s.insert(0, sChords)
            for linechord in fchord.readlines():
                linechordtotal += linechord
            realchordtotal = linechordtotal.split()
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                #print(len(chordTotal))
                #print(i)
                if(i < len(chordTotal)):
                    currentChord = chordTotal[i]#.encode('ansi')  # string to byte
                    #print(currentChord)
                    if(realchordtotal[i] != realchordtotal[i-1]):
                        realchord = realchordtotal[i].encode('ansi')
                        thisChord.addLyric(realchord.decode('utf-8'))
                    elif i == 0:
                        realchord = realchordtotal[i].encode('ansi')
                        thisChord.addLyric(realchord.decode('utf-8'))
                    else:
                        thisChord.addLyric(' ')
                    if (chordTotal[i] != 'n/a' and chordTotal[i] != 'n'):
                        thisChord.addLyric(chordTotal[i])
                    else:
                        thisChord.addLyric(' ')
                    if(chordpreTotal[i] != 'n/a' and chordpreTotal[i] != 'n'):
                        thisChord.addLyric(chordpreTotal[i])
                    else:
                        thisChord.addLyric(' ')
                    '''if(chordTotal[i] != chordpreTotal[i]):
                        if(chordTotal[i] != 'n/a' and chordTotal[i] != 'n'):
                            thisChord.addLyric(chordTotal[i])
                        if (chordpreTotal[i] != 'n/a'):
                            thisChord.addLyric(chordpreTotal[i])#.decode('utf-8'))  # byte to string
                    elif(chordTotal[i] != 'n/a' and chordTotal[i] != 'n'):
                        thisChord.addLyric(chordTotal[i])
                        thisChord.addLyric(chordpreTotal[i])'''
                else:
                    print('error')
                thisChord.closedPosition(forceOctave=4, inPlace=True)
            s.write('musicxml', fp=".\\predicted_result\\" + 'predicted_result_' + fn + 'non_chord_tone_' + sign + pitch + '.xml')

def translate_chord_name_into_music21(chordname):
    """
    translate chord name from Rameau into music21 version
    Split multi interpretation
    :param chordname:
    :return:
    """
    #print('original chord: ', chordname)
    multi = []
    for ii, item in enumerate(chordname):
        if(item == 'b'):
            if(ii != 0):
                if(chordname[ii - 1].isalpha()):  # this means flat!
                    chordname = chordname[:ii] + '-' + chordname [ii + 1:]
                else:
                    chordname = chordname[:ii] + 'B' + chordname[ii + 1:]
            #else:
                #print('$%^&*')
    #print('translated: ', chordname)
    chordname = chordname.replace('°', 'o')
    chordname = chordname.replace('ø', '/o')
    if chordname.find('/o') != -1 and chordname.find('/o7') == -1:  # must be half diminished 7th!
        chordname = chordname.replace('/o', '/o7')
    i = chordname
    if (i.find('nil') == -1 and i.find('+7+') == -1 and i.find('it') == -1 and i.find('.') == -1 and i.find('m7+') == -1
        and i.find('ee') == -1 and i.find('5+') == -1 and i.find('+6') == -1 and i.find('f#/o/a') == -1
        and i.find('f#c#s') == -1 and i.find('fis') == -1 and i.find('af') == -1 and i.find('d7f#') == -1
        and i !='7/f#' and i.find(']') == -1 and i.find('7M') == -1 and i != '7' and i.find('g#7-') == -1
        and i.find('es') == -1 and i.find('cis') == -1 and i.find('F#7A#') == -1 and i.find('c#b') == -1):
        if i.find('7+') != -1 :
            i = i.replace('7+','M7')
        #d = harmony.ChordSymbol(i)
        if i.find('oth') != -1:
            return i, 0, multi
        elif i.find(',') == -1:
            return i, 1, multi
        else:  # contain multiple interpretations, no need to make chord of music21
            multi = i.split(',')
            return i, 2, multi  # multiple but valid chords
    else:
        return i, 0, multi

def put_music21chord_into_musicXML(sign):
    '''
    Only put music21 chord into XML
    :param sign:
    :return:
    '''
    allowed_qualities = [[0,4,7], [0,3,6], [0,3,7], [0,4,8], [0,3,6,9], [0,3,6,10], [0,3,7,10],[0,4,7,10],[0,4,7,11]]
    for id, fn in enumerate(os.listdir(cwd)):
        # print(fn)
        if fn[-3:] == 'krn':
            if (os.path.isfile('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop''')):

                f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop', 'r')
                #fprediction = open('.\\predicted_result\\transposed_predicted_result_'+ fn[4:7] + '.txt', 'r')

            elif (
            os.path.isfile('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop.not', 'r')
                #fprediction = open('.\\predicted_result\\transposed_predicted_result_' + fn[4:7] + '.txt', 'r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            sChords = s.chordify()
            lineTotal = ''
            for line in f.readlines():
                line = get_chord_line(line, sign)
                lineNoInversion = get_chord_line(line, '0')
                lineTotal += line
                #lineTotalNoInversion += lineNoInversion
            chordTotal = lineTotal.split()
            sum_of_slices = len(sChords.recurse().getElementsByClass('Chord'))
            s.insert(0, sChords)
            cor_music21 = 0
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                #print(len(chordTotal))
                #print(i)
                if(i < len(chordTotal)):
                    currentChord = chordTotal[i].encode('ansi')  # string to byte
                    thisChord.addLyric(currentChord.decode('utf-8'))  # byte to string
                    music21_chord, valid_chord, multi_chord = translate_chord_name_into_music21(currentChord.decode('utf-8'))  #
                    # check music21 harmonic analysis
                    if chordTotal[i].find(',') != -1 :  # give a separate line for multiple interpretations
                        thisChord.addLyric(multi_chord)
                    else:
                        thisChord.addLyric(' ')
                        #cor_multi += 1
                    if valid_chord == 1:  # output the harmonic analysis of music21
                        #print('original chord in music21 readable format: ', music21_chord)
                        print(thisChord.pitchedCommonName)
                        print(thisChord.primeForm)
                        print(thisChord.normalForm)
                        print(thisChord.normalOrder)
                        print('----------------')
                        a = harmony.ChordSymbol(music21_chord)
                        if (thisChord.normalForm in allowed_qualities or thisChord.pitchedCommonName.find(
                                'incomplete') != -1
                                or ((thisChord.pitchedCommonName.find(
                                    'dominant') != -1 or thisChord.pitchedCommonName.find(
                                    'diminished') != -1 or thisChord.pitchedCommonName.find('major') != -1
                                     or thisChord.pitchedCommonName.find(
                                            'half-diminished') != -1 or thisChord.pitchedCommonName.find('minor') != -1)
                                    and thisChord.pitchedCommonName.find('seventh') != -1)): #or (music21_chord.lower() == currentChord.decode('utf-8').lower()):
                            #thisChord.addLyric(a.normalOrder)
                            #thisChord.addLyric(thisChord.normalOrder)
                            thisChord.addLyric('✓')
                            cor_music21 += 1
                        else:
                            #thisChord.addLyric(a.normalOrder)
                            #thisChord.addLyric(thisChord.normalOrder)
                            #thisChord.addLyric('music21 disagrees ')#thisChord.pitchedCommonName)
                            '''if(thisChord.normalForm in allowed_qualities or thisChord.pitchedCommonName.find('incomplete') != -1
                                    or ((thisChord.pitchedCommonName.find('dominant') != -1 or thisChord.pitchedCommonName.find('diminished') != -1 or thisChord.pitchedCommonName.find('major') != -1
                                    or thisChord.pitchedCommonName.find('half-diminished') != -1 or thisChord.pitchedCommonName.find('minor') != -1)
                                        and thisChord.pitchedCommonName.find('seventh') != -1)):
                                print(thisChord.primeForm)
                                thisChord.addLyric(thisChord.pitchedCommonName)
                            else:'''
                            thisChord.addLyric('ncts!')
                    elif valid_chord == 0:  # invalid chord, it must be different for Rameau and music21 analysis
                        thisChord.addLyric(thisChord.pitchedCommonName)
                thisChord.closedPosition(forceOctave=4, inPlace=True)
            thisChord.addLyric('num of slices:' + str(sum_of_slices))
            thisChord.addLyric('music21 acc:' + str(cor_music21 / sum_of_slices))
            #thisChord.addLyric('prediction acc:' + str(cor_prediction / sum_of_slices))
            #thisChord.addLyric('% of ambiguity:' + str(1 - cor_multi / sum_of_slices))
            s.write('musicxml', fp=".\\predicted_result\\" + fn[4:7] + 'music21.xml')
def put_chord_into_musicXML(sign):
    """

    :param string:
    :param string1:
    :param string2:
    :return:
    """
    for id, fn in enumerate(os.listdir(cwd)):
        # print(fn)
        if fn[-3:] == 'krn':
            if (os.path.isfile('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop''')):

                f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop', 'r')
                fprediction = open('.\\predicted_result\\transposed_predicted_result_'+ fn[4:7] + '.txt', 'r')

            elif (
            os.path.isfile('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop.not''')):
                f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop.not', 'r')
                fprediction = open('.\\predicted_result\\transposed_predicted_result_' + fn[4:7] + '.txt', 'r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            print(fn[4:7])
            sChords = s.chordify()
            lineTotal = ''
            lineTotalNoInversion = ''
            print(fprediction.readlines())
            for linepre in fprediction.readlines():

                linepre = get_chord_line(linepre, sign)
                lineTotal += linepre
            chordpreTotal = lineTotal.split()
            lineTotal = ''
            for line in f.readlines():
                line = get_chord_line(line, sign)
                lineNoInversion = get_chord_line(line, '0')
                lineTotal += line
                lineTotalNoInversion += lineNoInversion
            chordTotal = lineTotal.split()
            chordTotalNoInversion = lineTotalNoInversion.split()
            s.insert(0, sChords)
            sum_of_slices = len(sChords.recurse().getElementsByClass('Chord'))
            cor_music21 = 0
            cor_prediction = 0
            cor_multi = 0
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                #print(len(chordTotal))
                #print(i)
                if(i < len(chordTotal)):
                    currentChord = chordTotal[i].encode('ansi')  # string to byte
                    thisChord.addLyric(currentChord.decode('utf-8'))  # byte to string
                    music21_chord, valid_chord, multi_chord = translate_chord_name_into_music21(currentChord.decode('utf-8'))  #
                    # check music21 harmonic analysis
                    if chordTotal[i].find(',') != -1 :  # give a separate line for multiple interpretations
                        thisChord.addLyric(multi_chord)
                    else:
                        thisChord.addLyric(' ')
                        cor_multi += 1
                    if valid_chord == 1:  # output the harmonic analysis of music21
                        print('original chord in music21 readable format: ', music21_chord)
                        a = harmony.ChordSymbol(music21_chord)
                        if a.normalOrder == thisChord.normalOrder : #or (music21_chord.lower() == currentChord.decode('utf-8').lower()):
                            #thisChord.addLyric(a.normalOrder)
                            #thisChord.addLyric(thisChord.normalOrder)
                            thisChord.addLyric('✓')
                            cor_music21 += 1
                        else:
                            #thisChord.addLyric(a.normalOrder)
                            #thisChord.addLyric(thisChord.normalOrder)
                            thisChord.addLyric('music21 disagrees ')#thisChord.pitchedCommonName)
                    elif valid_chord == 0:  # invalid chord, it must be different for Rameau and music21 analysis
                        thisChord.addLyric(thisChord.pitchedCommonName)
                    else:  # multiple interpretations
                        for ii, item in enumerate(multi_chord):  # do not mess up with i! USe ii
                            #print('item is: ' , item)
                            a = harmony.ChordSymbol(item)
                            if a.normalOrder == thisChord.normalOrder:
                                thisChord.addLyric('✓')# + item)
                                cor_music21 += 1
                                break
                            else:
                                if ii == len(multi_chord) - 1:  # if all the interpretations do not match music21 analysis
                                    # output the difference
                                    thisChord.addLyric('music21 disagrees ')
                    #print('ori', currentChord.decode('utf-8'))
                    if currentChord.decode('utf-8').lower().find(chordpreTotal[i].lower()) == -1 :  # as long as the first letter is the same, the chords are the same
                        #print(chordTotalNoInversion[i].lower(), chordpreTotal[i].lower())
                        #print('ori pre:', chordpreTotal[i])
                        currentChord = chordpreTotal[i].encode('ansi')  # string to byte
                        music21_chord1, valid_chord1, multi_chord1 = translate_chord_name_into_music21(chordpreTotal[i])
                        music21_chord, valid_chord, multi_chord = translate_chord_name_into_music21(
                            chordTotal[i].encode('ansi').decode('utf-8'))
                        #if music21_chord.find('B-M7/F') != -1:
                            #input('debug B-M7/F')
                        #print("rameau chord", music21_chord)
                        #print("prediction chord", music21_chord1)
                        #print("valid_chord", valid_chord)
                        #print("valid_chord1", valid_chord1)
                        if (valid_chord1 == valid_chord == 1) and (harmony.ChordSymbol(music21_chord1).normalOrder == harmony.ChordSymbol(music21_chord).normalOrder):
                            print('where a# is the same with bb!')
                            thisChord.addLyric('✓')# + music21_chord1)
                            cor_prediction += 1
                        #print("pre (byte): ", currentChord)
                        #print("pre: ", currentChord.decode('utf-8'))
                        else:
                            if chordTotal[i][0].lower() == chordpreTotal[i][0].lower() :
                                thisChord.addLyric('✓')# + music21_chord1)
                                cor_prediction += 1
                            else:
                                thisChord.addLyric('prediction disagrees: ' + chordpreTotal[i])
                        #thisChord.addLyric(currentChord.decode('utf-8'))  # byte to string
                    else:
                        if(len(multi_chord) > 0):
                            thisChord.addLyric('✓')# + chordpreTotal[i])
                            cor_prediction += 1
                        else:
                            thisChord.addLyric('✓')
                            cor_prediction += 1
                else:
                    print('error')
                thisChord.closedPosition(forceOctave=4, inPlace=True)
            thisChord.addLyric('num of slices:' + str(sum_of_slices))
            thisChord.addLyric('music21 acc:' + str(cor_music21/sum_of_slices))
            thisChord.addLyric('prediction acc:' + str(cor_prediction / sum_of_slices))
            thisChord.addLyric('% of ambiguity:' + str(1 - cor_multi / sum_of_slices))
            s.write('musicxml', fp="C:\\Users\\User\\PycharmProjects\\harmonic_analysis\\predicted_result\\" + fn[4:7] + '.xml')
def put_annotation_into_musicXML(sign, multi):
    """
    Function where the annotations of Rameau are shown, and indicates 1-3 voice movements for annotator to label in two
    styles.
    :param string:
    :param string1:
    :param string2:
    :return:
    """
    for id, fn in enumerate(os.listdir(cwd)):
        # print(fn)
        if fn[-3:] == 'krn':
            if (os.path.isfile('.\\useful_chord_symbols\\translated_' + fn[4:7] + '.pop''')):
                if multi == 0:
                    f = open('.\\useful_chord_symbols\\translated_'+ fn[4:7] + '.pop', 'r')
                else:
                    f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop', 'r')
                #fprediction = open('.\\predicted_result\\transposed_predicted_result_'+ fn[4:7] + '.txt', 'r')
            elif (
            os.path.isfile('.\\useful_chord_symbols\\translated_' + fn[4:7] + '.pop.not''')):
                if multi == 0:
                    f = open('.\\useful_chord_symbols\\translated_'+ fn[4:7] + '.pop.not', 'r')
                else:
                    f = open('.\\useful_chord_symbols\\Multi_interpretation_156\\translated_multi' + fn[4:7] + '.pop.not', 'r')
                #fprediction = open('.\\predicted_result\\transposed_predicted_result_' + fn[4:7] + '.txt', 'r')
            else:
                continue  # skip the file which does not have chord labels
            s = converter.parse(cwd + fn)
            print(fn[4:7])
            sChords = s.chordify()
            lineTotal = ''
            lineTotalNoInversion = ''
            #for linepre in fprediction.readlines():
                #linepre = get_chord_line(linepre, sign)
                #lineTotal += linepre
            #chordpreTotal = lineTotal.split()
            lineTotal = ''
            for line in f.readlines():
                line = get_chord_line(line, sign)
                lineNoInversion = get_chord_line(line, '0')
                lineTotal += line
                lineTotalNoInversion += lineNoInversion
            chordTotal = lineTotal.split()
            chordTotalNoInversion = lineTotalNoInversion.split()
            sChords.volume = volume.Volume(velocity=90)
            s.insert(0, sChords)
            sum_of_slices = len(sChords.recurse().getElementsByClass('Chord'))
            cor_music21 = 0
            cor_prediction = 0
            cor_multi = 0
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                #print(len(chordTotal))
                #print(i)
                if(i < len(chordTotal)):
                    currentChord = chordTotal[i].encode('ansi')  # string to byte
                    thisChord.addLyric(currentChord.decode('utf-8'))  # byte to string
                    music21_chord, valid_chord, multi_chord = translate_chord_name_into_music21(currentChord.decode('utf-8'))  #
                    # check music21 harmonic analysis
                    if chordTotal[i].find(',') != -1 :  # give a separate line for multiple interpretations
                        thisChord.addLyric(multi_chord)
                    else:
                        if(i == 0):
                            thisChord.addLyric('Multiple interpretations')
                        else:
                            thisChord.addLyric(' ')
                        cor_multi += 1
                    if(i == 0):
                        thisChord.addLyric('NO. of PC changes: ')
                        thisChord.addLyric('Maximally melodic: ')
                        thisChord.addLyric('Maximally harmonic: ')
                        lastChord = thisChord
                    else:
                        common_pitch = 0
                        print("current chord: ", len(thisChord.pitchNames), "pitch: ", thisChord.pitchNames)
                        print("previous chord: ", len(lastChord.pitchNames) , "pitch: ", lastChord.pitchNames)
                        for ii, item in enumerate(thisChord.orderedPitchClasses):
                            if(item in lastChord.orderedPitchClasses):
                                common_pitch += 1
                        if(common_pitch == 3):
                            print('debug')
                        if(len(thisChord.orderedPitchClasses) >= len(lastChord.orderedPitchClasses)):
                            thisChord.addLyric(str(len(thisChord.orderedPitchClasses) - common_pitch) + '/' + str(len(thisChord.orderedPitchClasses)))
                            if(len(thisChord.orderedPitchClasses) - common_pitch == 0 or common_pitch == 0):
                                thisChord.addLyric('.')
                                thisChord.addLyric('.')
                            else:
                                thisChord.addLyric('_')
                                thisChord.addLyric('_')
                        else:
                            thisChord.addLyric(str(len(lastChord.orderedPitchClasses) - common_pitch) + '/' + str(len(lastChord.orderedPitchClasses)))
                            if (len(lastChord.orderedPitchClasses) - common_pitch == 0 or common_pitch == 0):
                                thisChord.addLyric('.')
                                thisChord.addLyric('.')
                            else:
                                thisChord.addLyric('_')
                                thisChord.addLyric('_')

                        lastChord = thisChord
                thisChord.closedPosition(forceOctave=4, inPlace=True)
            s.write('musicxml', fp=".\\predicted_result\\" + "annotation" + fn[4:7] + '.xml')
if __name__ == "__main__":
    # Get input features
    #sign = input("do you want inversions or not? 1: yes, 0: no")
    #output_dim =  input('how many kinds of chords do you want to calculate?')
    #put_music21chord_into_musicXML('1')
    #put_annotation_into_musicXML('1', 1)
    input = '.\\bach-371-chorales-master-kern\\kern\\'
    output = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Melodic\\'
    put_non_chord_tone_into_musicXML(input, output, '1')
    #put_chord_into_musicXML(sign, multi):