from music21 import *
import os
import sys
#from get_input_and_output import get_chord_list
def translate_chord_name_into_music21(chordname):
    """
    translate chord name from Rameau into music21 version
    :param chordname:
    :return:
    """
    print('original chord: ', chordname)
    for i, item in enumerate(chordname):
        if(item == 'b'):
            if(i != 0):
                if(chordname[i - 1].isalpha()):  # this means flat!
                    chordname = chordname[:i] + '-' + chordname [i + 1:]
                else:
                    chordname = chordname[:i] + 'B' + chordname[i + 1:]
            else:
                print('$%^&*')
    print('translated: ', chordname)
    chordname = chordname.replace('°', 'o')
    chordname = chordname.replace('ø', '/o')
    return chordname
def get_chord_tone(i,outputdim):
    """
    return chord tone in a pitch class vector
    :param chord:
    :return:
    """
    chordtone = [0] * (outputdim + 1)
    currentChord = i.encode('ansi')
    i = currentChord.decode('utf-8')
    i = translate_chord_name_into_music21(i)
    if (i.find('nil') == -1 and i.find('+7+') == -1 and i.find('it') == -1 and i.find('.') == -1 and i.find('m7+') == -1
        and i.find('ee') == -1 and i.find('5+') == -1 and i.find('+6') == -1 and i.find('f#/o/a') == -1
        and i.find('f#c#s') == -1 and i.find('fis') == -1 and i.find('af') == -1 and i.find('d7f#') == -1
        and i !='7/f#' and i.find(']') == -1 and i.find('7M') == -1 and i != '7' and i.find('g#7-') == -1):
        d = harmony.ChordSymbol(i)
        for j in d.pitchClasses:
            chordtone[j] = 1
    else:
        chordtone[outputdim] = 1  # 'other' category
    return chordtone
'''if __name__ == "__main__":
    s = converter.parse('001.xml')
    f = open('chordtranslation.txt', 'w')
    sChords = s.parts[-1]
    lyric = sChords.lyrics()
    print(lyric)
    lyric[1]
    list_of_chords = get_chord_list(300, '1')
    for i in list_of_chords:
        currentChord = i.encode('ansi')
        i = currentChord.decode('utf-8')
        i = translate_chord_name_into_music21(i)
        if(i.find('nil')==-1 and i.find('+7+')==-1 and i.find('it')==-1 and i.find('.')==-1 and i.find('m7+')==-1
           and i.find('ee')==-1 and i.find('5+')==-1 and i.find('+6') == -1 and i.find('f#/o/a') == -1
           and i.find('f#c#s') == -1 and i.find('fis') == -1 and i.find('af') == -1 and i.find('d7f#') == -1
           and i.find('7/f#') == -1 and i.find(']') == -1 and i.find('7M') == -1 and i != '7' and i.find('g#7-') == -1):
            d = harmony.ChordSymbol(i)
            print(i, file=f)
            print(d.pitches, file=f)
            #print(d.pitchClasses)'''

