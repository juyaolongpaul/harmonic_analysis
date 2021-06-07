import xml.etree.cElementTree as ET
import os
from music21 import *
import re
import codecs
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from collections import Counter
from code_snippet import put_chords_into_files, key_invariant_pairs
from get_input_and_output import get_pitch_class_for_four_voice, get_bass_note, get_FB, colllapse_interval, is_suspension, get_next_note, get_previous_note, contain_continuo_voice, remove_instrumental_voices, contain_chordify_voice
from mido import MetaMessage, MidiFile


f_sus = open('suspension.txt', 'w')


def get_actual_figures(bass, sonority, actual_FB, key, duplicate='N'):
    aInterval = interval.Interval(noteStart=bass, noteEnd=sonority)
    if int(aInterval.name[1:]) % 7 == 0:
        FB_desired = '7'
    elif '9' == aInterval.name[1] or '8' == aInterval.name[1]:
        FB_desired = aInterval.name[1:]
    else:
        FB_desired = str(int(aInterval.name[1:]) % 7)
    if FB_desired == '1':
          # only non-bass can be translated into 8
        FB_desired = '8'

    # Sometimes it can share the same pitch class, and in this case,
    # it will give two identical FB, in this case, we only need one

    if sonority.pitch.accidental is not None:
        if not any(sonority.pitch.pitchClass == each_scale.pitchClass for each_scale in key.pitches):
            if FB_desired == '3':
                actual_FB = add_FB_result(actual_FB,
                                          sonority.pitch.accidental.unicode, duplicate)
            else:
                if not (aInterval.name[0] == 'P' and FB_desired == '8'):
                    actual_FB = add_FB_result(actual_FB,
                                              sonority.pitch.accidental.unicode + FB_desired, duplicate)
                else:
                    actual_FB = add_FB_result(actual_FB,
                                              FB_desired, duplicate)  # the exception is where continuo and bass are both raised, we need to output 8 not #8!

        else:
            actual_FB = add_FB_result(actual_FB,
                                      FB_desired, duplicate)
    else:
        actual_FB = add_FB_result(actual_FB,
                                  FB_desired, duplicate)
    return actual_FB

def add_FB_result(actual_FB, result, duplicate='N'):
    """
    Modular function to add the FB annotation. If already exist, skip this part which will add a duplicated one, since
    there can be situations where multiple voices share the same pitch class
    :param thisChord:
    :param actual_FB:
    :return:
    """
    if duplicate == 'N':  # don't want duplicates
        if result not in actual_FB:
            # thisChord.addLyric(result)
            actual_FB.append(result)
    else:  # output duplicates
        actual_FB.append(result)
    return actual_FB


def translate_FB_as_lyrics(number, suffix, prefix, extension):
    """
    Translate FB in lists into string added to lyrics
    :param number:
    :param fix:
    :param duration:
    :return:
    """

    if prefix == 'sharp':
        prefix_sign = '#'
    elif prefix == 'flat':
        prefix_sign = 'b'
    elif prefix == 'natural':
        prefix_sign = 'n'
    else:
        if prefix != '':
            print(prefix)
            input('this is the prefix you have not considered yet!')
        else:
            prefix_sign = prefix
    if suffix == 'backslash' or suffix == 'cross':
        suffix_sign = '#'
    elif suffix == 'natural':
        suffix_sign = 'n'
    elif suffix == 'sharp':
        suffix_sign = '#'
    elif suffix == 'flat':
        suffix_sign = 'b'
    elif suffix == 'slash':
        suffix_sign = 'b'
    else:
        if suffix != '':
            print(suffix)
            input('this is the suffix you have not considered yet!')
        else:
            suffix_sign = suffix
    if extension == '': # no extension sign
        return prefix_sign+suffix_sign+number
    else:
        return prefix_sign + suffix_sign + number + '_'  # plus the continuation line is found


def add_FB_content(j, each_fb, i, text):
    """
    Template to add figured bass content to lyrics
    :param j:
    :param each_fb:
    :param i:
    :return:
    """

    if i < len(each_fb['number']):
        if 'duration' in each_fb:
            if text.text is None:
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i], each_fb['extension'][i]) + '+' + each_fb['duration']
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i], each_fb['extension'][i]) + '+' + each_fb['duration']
        else:
            if text.text is None:
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i], each_fb['extension'][i])
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i], each_fb['extension'][i])
        return text
    else:
        if j == 0: # this is when a figured bass does not have as much layers as the following ones
            text.text = ''
        return text

def add_FB_to_lyrics(note, fig):
    """
    Add figured bass info underneath as lyrics for the current bass note
    :param ele:
    :param fig:
    :return:
    """
    number_of_layers = 0
    for i, each_fb in enumerate(fig):
        if number_of_layers < len(each_fb['number']):
            number_of_layers = len(each_fb['number'])
    for i in range(number_of_layers):  # i means the current number of layer, horizontally
        #print('numer of layers', i)
        lyric = ET.SubElement(note, 'lyric', {'number':str(i + 1)})  # create lyric sub element
        text = ET.SubElement(lyric, 'text')
        for j, each_fb in enumerate(fig):  # j means the ID of figures for the bass in the current layer
            if j == 0:
                text = add_FB_content(j, each_fb, i, text)
            else:
                text.text += ','
                text = add_FB_content(j, each_fb, i, text)


def adding_XXXfix(each_FB_digit, name, single_XXXfix):
    XXXfix = each_FB_digit.find(name)
    if XXXfix is not None:
        if name == 'extend':
            single_XXXfix.append(each_FB_digit.find(name).attrib)
        else:
            single_XXXfix.append(each_FB_digit.find(name).text)
    else:
        single_XXXfix.append('')
    return single_XXXfix


def decode_FB_from_lyrics(lyrics):
    """
    Decoding the FB from the bassline line into list dictionary represetnation
    :param lyrics:
    :return:
    """
    fig = []
    number_of_layers = len(lyrics)
    for i ,each_layer in enumerate(lyrics):
        figure_duration = each_layer.text.split(',')
        #print (figure_duration)
        for j, each_figure_duration in enumerate(figure_duration):
            if i == 0:
                each_figure_dic = {}
                if '+' in each_figure_duration:
                    each_figure = each_figure_duration.split('+')[0]
                    each_duration = each_figure_duration.split('+')[1]
                    temp = []
                    temp.append(each_figure)
                    each_figure_dic['number'] = temp
                    each_figure_dic['duration'] = each_duration
                else:  # just figures
                    each_figure = each_figure_duration
                    temp = []
                    temp.append(each_figure)
                    each_figure_dic['number'] = temp
                fig.append(each_figure_dic)
            else:  # add to existing FB dictionary
                if '+' in each_figure_duration:
                    each_figure = each_figure_duration.split('+')[0]
                    each_duration = each_figure_duration.split('+')[1]
                else:
                    each_figure = each_figure_duration
                fig[j]['number'].append(each_figure)

    return fig


def is_legal_chord(chord_label):
    """
    Judge whether the current chord label is legal or not based on the sonority
    :param chord_label:
    :return:
    """
    allowed_chord_quality = ['incomplete major-seventh chord', 'major seventh chord',
                             'incomplete minor-seventh chord', 'minor seventh chord',
                             'incomplete half-diminished seventh chord', 'half-diminished seventh chord',
                             'diminished seventh chord',
                             'incomplete dominant-seventh chord', 'dominant seventh chord',
                             'major triad',
                             'minor triad',
                             'diminished triad',
                             'augmented triad', '-interval class 3', '-interval class 4']
    try:
        chord_name = chord_label.pitchedCommonName
    except:
        return None
    flag = 0 # 1 means it is the following three
    if chord_name.find('-interval class 3') != -1:
        chord_name = chord_name.replace('-interval class 3', '') + 'm'
        flag = 1
    elif chord_name.find('-interval class 4') != -1:
        chord_name = chord_name.replace('-interval class 4', '')
        flag = 1
    # elif chord_name.find('-interval class 5') != -1:
    #     chord_name = chord_name.replace('-interval class 5', '')
    #     flag = 1
    chord_name = re.sub(r'\d', '', chord_name)  # remove all the octave information
    if any(each in chord_name for each in allowed_chord_quality):
        if harmony.chordSymbolFigureFromChord(chord_label).find(
                'Identified') != -1:  # harmony.chordSymbolFigureFromChord cannot convert pitch classes into chord name sometimes, and the examples are below
            # print('debug')
            # print('debug')
            if chord_name.find(
                    '-diminished triad') != -1:  # chord_label.pitchedCommonName is another version of the chord name, but usually I cannot use it to get harmony.ChordSymbol to get pitch classes, so I translate these cases which could be processed by harmony.ChordSymbol later on
                chord_name = chord_name.replace('-diminished triad', 'o')  # translate to support
            elif chord_name.find('-incomplete half-diminished seventh chord') != -1:
                chord_name = chord_name.replace('-incomplete half-diminished seventh chord',
                                                                   '/o7')  # translate to support
            elif chord_name.find('-incomplete minor-seventh chord') != -1:
                chord_name = chord_name.replace('-incomplete minor-seventh chord',
                                                                   'm7')  # translate to support
            elif chord_name.find('-incomplete major-seventh chord') != -1:
                chord_name = chord_name.replace('-incomplete major-seventh chord',
                                                                   'M7')  # translate to support
            elif chord_name.find('-incomplete dominant-seventh chord') != -1:
                chord_name = chord_name.replace('-incomplete dominant-seventh chord',  # TODO: fix the octave inclusion issue
                                                                   '7')  # translate to support
            elif chord_name.find('-major triad') != -1:  # (e.g., E--major triad) in  279 slice 33
                chord_name = chord_name.replace('-major triad', '')  # translate to support
            elif chord_name.find(
                    '-dominant seventh chord') != -1:  # (e.g., E--major triad) in  279 slice 33
                chord_name = chord_name.replace('-dominant seventh chord',
                                                                   '7')  # translate to support
            elif chord_name.find('-half-diminished seventh chord') != -1:
                chord_name = chord_name.replace('-half-diminished seventh chord',
                                                                   '/o7')  # translate to support
            elif chord_name.find('-minor-seventh chord') != -1:
                chord_name = chord_name.replace('-minor-seventh chord', 'm7')  # translate to support
            elif chord_name.find('-major-seventh chord') != -1:
                chord_name = chord_name.replace('-major-seventh chord', 'M7')  # translate to support
            else:
                chord_name = chord_name  # Just in case the function cannot accept any names (e.g., E--major triad)
        else:
            if chord_name.find(
                    '-incomplete dominant-seventh chord') != -1:  # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                chord_name = re.sub(r'/[A-Ga-g][b#-]*', '',
                                    chord_name.replace('-incomplete dominant-seventh chord',
                                                                          '7'))  # remove 'add' part
            elif chord_name.find(
                    '-incomplete major-seventh chord') != -1:  # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                chord_name = re.sub(r'/[A-Ga-g][b#-]*', '',
                                    chord_name.replace('-incomplete major-seventh chord',
                                                                          'M7'))  # remove 'add' part
            elif harmony.chordSymbolFigureFromChord(chord_label).find(
                    'add') != -1:  # contains "add" which does not work for harmony.ChordSymbol, at 095
                if 'half-diminished seventh chord' not in chord_name:
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[
                                                                :harmony.chordSymbolFigureFromChord(chord_label).find(
                                                                    'add')])  # remove 'add' part
                else:
                    chord_name = chord_name.replace('-half-diminished seventh chord',
                                                    '/o7')  # translate to support
            # elif harmony.chordSymbolFigureFromChord(chord_label).find('power') != -1: # assume power alone as major triad
            #     chord_label_list.append(
            #         re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[
            #                                        :harmony.chordSymbolFigureFromChord(chord_label).find(
            #                                            'power')]))  # remove 'add' part
            elif harmony.chordSymbolFigureFromChord(chord_label).find('dim') != -1:
                chord_name = re.sub(r'/[A-Ga-g][b#-]*', '',
                                    harmony.chordSymbolFigureFromChord(chord_label).replace('dim', 'o'))
            else:
                chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(
                    chord_label))  # remove inversions, notice that half diminished also has /!
            # the line above is the most cases, where harmony.chordSymbolFigureFromChord can give a chord name for the pitch classes, and Bdim is generated by this!
        return chord_name.replace('maj7', 'M7')
    elif flag == 1:
        return chord_name.replace('maj7', 'M7')
    else:
        return None


def translate_FB_into_PC(all_key_pitches, bass, each_figure, chord_pitch):

    for each_key_pitch in all_key_pitches:
        aInterval = interval.Interval(noteStart=bass, noteEnd=each_key_pitch)
        # if bass.pitch.midi > each_key_pitch.pitch.midi:
        #         #     input('bass is even higher than the key pitch!')
        collapsed_fig = colllapse_interval(aInterval.name[1:])
        if collapsed_fig == each_figure:
            chord_pitch.append(each_key_pitch.name)
            break
        elif len(each_figure) == 2 and each_figure[1] == collapsed_fig:  # the figure has accidentals
            if each_figure[0] == 'b':
                if each_key_pitch.accidental is not None:
                    if each_key_pitch.accidental.modifier != '-':  # this note is not flatten in the signature
                        chord_pitch.append(each_key_pitch.transpose('d1').name)
                    else:  # otherwise, this note is already flatten, we don't need to flat it again
                        chord_pitch.append(each_key_pitch)
                else: # this not is natural in the key scale, but FB wants to flatten it
                    chord_pitch.append(each_key_pitch.transpose('d1').name)
                break
            elif each_figure[0] == '#':
                if each_key_pitch.accidental is not None:
                    if each_key_pitch.accidental.modifier != '#':
                        chord_pitch.append(each_key_pitch.transpose('a1').name)
                    else:
                        chord_pitch.append(each_key_pitch)
                else: # this not is natural in the key scale, but FB wants to sharpen it
                    chord_pitch.append(each_key_pitch.transpose('a1').name)
                break
            elif each_figure[0] == 'n':
                if each_key_pitch.accidental is not None:
                    if each_key_pitch.accidental.modifier == '-':
                        chord_pitch.append(each_key_pitch.transpose('a1').name)
                        break
                    elif each_key_pitch.accidental.modifier == '#':
                        chord_pitch.append(each_key_pitch.transpose('d1').name)
                        break
                else:
                    chord_pitch.append(each_key_pitch)


def add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, number):
    if number not in implied_fig_collapsed_no_accidental:
        implied_fig_collapsed.append(number)
        implied_fig_collapsed_no_accidental.append(number)

def get_chord_tone(thisChord, fig, s, a_discrepancy, a_slice_discrepancy, condition='N'):
    """
    Function to determine which sonorities are chord tones or not based on the given FB
    :param pitchNames:
    :param fig:
    :return:
    """

    # There can be three cases: (1) FB indicates all the sonorities (2) FB only indicate part of the sonority
    # (3) FB indicates some notes not in the sonorities
    # For now label (2) as ??, for (3), look at the next slice, and the FB should be there, if not, output ?!
    # So far, we collapse both figured bass and intervals, which means 9 and 2 are interchangable, etc.
    chord_pitch = []
    discrepancies = []
    #print('current fig is', fig)
    if fig != [' '] and fig != '':
        fig_collapsed = colllapse_interval(fig)
    key_measure_ID = []
    keys = []
    for ID, each_measure in enumerate(s.parts[0].recurse().getElementsByClass('Measure')):
        if each_measure.keySignature is not None:
            key_measure_ID.append(ID)
            keys.append(each_measure.keySignature)
    if len(key_measure_ID)== 1: # only one key signature
        altered_pitch = keys[0].alteredPitches
    else:
        input('key signature changed, and we need to deal with it!')
    all_key_pitches = list(altered_pitch)
    natural_pitch  = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    for each_natural_pitch in natural_pitch:
        if any(each_natural_pitch == each_altered_pitch.step for each_altered_pitch in altered_pitch): # this pitch is altered, no need to add it
            continue
        else:
            all_key_pitches.append(pitch.Pitch(each_natural_pitch))
    # PC
    if condition == 'N': # this means we need to choose which ones are CT based on the FB
        mark = ''  # indicating weird slices
        intervals = []
        pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
        bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')
        # first, we have algorithm A, which translates chord only based on the bassline and the figures
        # all the slices with figured bass will be examined: (1) all the implied figures will be added,
        # and (2) we obtain the pitch class indicated by all these figures, and consider them as chord tones
        # if they form a legal chord, then the chord label is output by the algorithm
        # Step (1)
        implied_fig_collapsed = list(fig_collapsed)
        # implied_fig_collapsed.append('1')  # add bass as chord tone
        chord_pitch.append(bass.pitch.name)  # the bass is chord tone!
        # in case "n", "#" and "b", I want to append "3" to them to translate chord tones based on the figure
        for ID, each_figure in enumerate(implied_fig_collapsed):
            if 'n' == each_figure or 'b' == each_figure or '#' == each_figure:
                implied_fig_collapsed[ID] += '3'
        fig_collapsed_no_accidental = []
        for each_figure in fig_collapsed:
            if 'n' in each_figure or 'b' in each_figure or '#' in each_figure:
                if len(each_figure) == 1:
                    fig_collapsed_no_accidental.append('3')
                else:
                    fig_collapsed_no_accidental.append(each_figure[1:])
            else:
                fig_collapsed_no_accidental.append(each_figure)
        implied_fig_collapsed_no_accidental = list(fig_collapsed_no_accidental)
        # the one above is used when "42" becomes "642", and we don't want to add a "5"  to it
        # the following rules also incoporate the ones in Arnold (1931) pp. 263 table
        if implied_fig_collapsed_no_accidental == ['2'] and '9' not in fig:
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '4')
        if implied_fig_collapsed_no_accidental == ['4', '3'] or implied_fig_collapsed_no_accidental == ['3', '4'] \
                or implied_fig_collapsed_no_accidental ==['4', '2'] or implied_fig_collapsed_no_accidental ==['2', '4']\
                or (implied_fig_collapsed_no_accidental == ['2'] and '9' not in fig):
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '6')
        if '6' not in implied_fig_collapsed_no_accidental:
            if fig_collapsed != ['#4']: # in this exception, 5 should not be added
                add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '5')
        if '4' not in implied_fig_collapsed_no_accidental and ('2' not in implied_fig_collapsed_no_accidental or '9' in fig):
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '3')
        if fig_collapsed == ['#4']: # if only #4, 6 and 2 need to be added
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '6')
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '2')
        # if fig_collapsed == ['b5']:
        #     add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '6')
        if fig_collapsed == ['b6', '#4'] or fig_collapsed == ['#4', 'b6']:
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '2')
        if fig_collapsed == ['2', '7'] or fig_collapsed == ['7', '2']:
            add_implied_FB(implied_fig_collapsed_no_accidental, implied_fig_collapsed, '4')
        # remove duplicated figures
        implied_fig_collapsed = list(dict.fromkeys(implied_fig_collapsed))

        print('implied figures are', implied_fig_collapsed)
        # print('debug')
        for each_figure in implied_fig_collapsed:
            translate_FB_into_PC(all_key_pitches, bass, each_figure, chord_pitch)
        # chord_pitch_class = []
        # for each_chord_pitch in chord_pitch:
        #     chord_pitch_class.append(each_chord_pitch.pitchClass)
        # for i, each_chord_pitch_class enumerate(chord_pitch_class):

        # chord_pitch = list(dict.fromkeys(chord_pitch)) # if there are duplicated pitches, the chord won't generate!
        for i, sonority in enumerate(thisChord._notes):
            # print(note)
            # print('bass', bass)
            if sonority.pitch.midi < bass.pitch.midi: # there is voice crossing, so we need to transpose the bass an octave lower, marked as '@'
                #mark += '@'
                bass_lower = bass.transpose(interval.Interval(-12))
                aInterval = interval.Interval(noteStart=bass_lower, noteEnd=sonority)
            else:
                aInterval = interval.Interval(noteStart=bass, noteEnd=sonority)
            colllapsed_interval = colllapse_interval(aInterval.name[1:])
            intervals.append(colllapsed_interval)
            # TODO: there might be cases where we need to check whether there is a real 9, or just a 2. In this case we cannot check
            # if ('3' in colllapsed_interval and '4' not in fig_collapsed_no_accidental and ('2' not in fig_collapsed_no_accidental or '9' in fig)) or ('5' in colllapsed_interval and '6' not in fig_collapsed_no_accidental) or '1' in colllapsed_interval:
            #     pass # now we don't reply on the surface to determine which figure is implied.
            #     # It is directly down above based on figured bass
            # elif any(colllapsed_interval in each for each in fig_collapsed_no_accidental):
            #     pass
            # elif colllapsed_interval == '6' and (fig_collapsed_no_accidental == ['4', '3'] or fig_collapsed_no_accidental == ['4', '2'] or fig_collapsed_no_accidental == ['2']):  # TODO: check if there is 246 case happen
            #     pass
            # elif colllapsed_interval == '4' and fig_collapsed_no_accidental == ['2']:
            #     pass
            # else: # sonority not in the FB
            #     mark += '??'
            #     a_discrepancy.append('??' + colllapsed_interval)
            #     discrepancies.append('??' + colllapsed_interval)
        for each_figure in fig_collapsed:
            if each_figure == '' or '_' in each_figure:
                continue
            if each_figure[-1] not in intervals:  # FB not in sonorities
                if each_figure in ['n', '#', 'b'] and '3' in intervals: # this is an exception
                    mark += ''
                else:
                    mark += '?!' # add the note indicated by FB as chord tone
                    translate_FB_into_PC(all_key_pitches, bass, each_figure, chord_pitch)
                    a_discrepancy.append(each_figure)
                    # now we don't need to use ?! to note
                    # since we don't consider ?? anymore, and there is only one type of discrepancies
                    discrepancies.append(each_figure)
        if mark == '':
            thisChord.addLyric(' ')
        else:
            discrepancies = list(dict.fromkeys(discrepancies))
            thisChord.addLyric(str(discrepancies).strip('[]'))
            a_slice_discrepancy.append(str(discrepancies).strip('[]'))
        return chord_pitch, mark
    else:
        thisChord.addLyric(' ')
        for each_pitch in thisChord.pitchNames:
            chord_pitch.append(each_pitch)
        return chord_pitch, ''


def add_chord(thisChord, chordname):
    """
    Remove unwanted symbolic from the chord name
    :param thisChord:
    :param chordname:
    :return:
    """
    if thisChord.lyrics[-1].text.replace('?', '').replace('!', '') != chordname.replace('?', '').replace('!', ''):
        # we don't want duplicated chord labels!
        if chordname[-1] == '-':  # This is the bug music21 has: for chords with flats, it will fail to add as lyrics
            thisChord.addLyric(chordname.replace('?', '').replace('!', '') + '*')  # adding a casual sign (then remove it) to avoid this bug
            thisChord.lyrics[-1].text = thisChord.lyrics[-1].text.replace('*', '')
        else:
            thisChord.addLyric(chordname.replace('?', '').replace('!', ''))  # get rid of all the ? and !
            # since it is already indicated with the figure


def label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr, sus_type, a_suspension):
    """
    Modular function to label suspension
    :param ptr:
    :param ptr2:
    :param s:
    :param sChord:
    :param voice_number:
    :param thisChord:
    :return:
    """
    if is_suspension(ptr, ptr2, s, sChord, voice_number, sus_type):
        thisChord.style.color = 'pink'
        print(str(ptr), file=f_sus)
        if ptr + ptr2 in suspension_ptr: # this case, it is double suspension
            if len(a_suspension)>1:
                if sus_type != a_suspension[-1]: # without this, we can have two 4-3 suspensions in one slice
                    a_suspension[-1] += sus_type
        else: # otherwise, it is only one suspension in this slice
            a_suspension.append(sus_type)
        if ptr + ptr2 not in suspension_ptr: # without this the first 6-5 suspension at m. 6 in BWV 72.06
            # will have two entries in this list
            suspension_ptr.append(ptr + ptr2)
    return suspension_ptr, a_suspension


def replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr, mark, s):
    if pitch_four_voice[-1].pitch.pitchClass == pitch_four_voice_next[-1].pitch.pitchClass or int(
            thisChord.beat) == int(sChord.recurse().getElementsByClass('Chord')[ptr + 1].beat):
        # same bass pitch class or different basses but same beat (362 mm.2 last)
        next_chord_pitch, next_mark = get_chord_tone(sChord.recurse().getElementsByClass('Chord')[ptr + 1], '', s, [], [],
                                                     'Y')  ## TODO: shouldn't we give the actual FB to this function?
        ## TODO: should we also consider the mark for the next chord in some ways?
        next_chord_label = chord.Chord(next_chord_pitch)
        next_chord_name = is_legal_chord(next_chord_label)
        if next_chord_name:
            add_chord(thisChord, mark + next_chord_name)
        else:
            replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr + 1, mark, s)
            # recursively looking for the next chord
    elif len(pitch_four_voice[-1].beams.beamsList) > 0 and len(pitch_four_voice_next[-1].beams.beamsList) > 0:
        if pitch_four_voice[-1].beams.beamsList[0].type == 'start' and pitch_four_voice_next[-1].beams.beamsList[
            0].type == 'stop':
            next_chord_pitch, next_mark = get_chord_tone(  # TODO: factorize this section of code
                sChord.recurse().getElementsByClass('Chord')[ptr + 1], '', s, [], [],
                'Y')  ## TODO: shouldn't we give the actual FB to this function?
            ## TODO: should we also consider the mark for the next chord in some ways?
            next_chord_label = chord.Chord(next_chord_pitch)
            next_chord_name = is_legal_chord(next_chord_label)
            if next_chord_name:
                add_chord(thisChord, mark + next_chord_name)
            else:
                replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr + 1, mark, s)
                # recursively looking for the next chord
        else:
            add_chord(thisChord, ' ' + mark)

    else:
        add_chord(thisChord, ' ' + mark)


def suspension_processing(fig, thisChord, bass, sChord, fig_collapsed, ptr, s, a_suspension, suspension_ptr):
    if '7' in fig or '6' in fig or '4' in fig or '2' in fig or '9' in fig:  # In these cases, examine whether this note is a suspension or not
        # if thisChord.measureNumber == 12:
        #     print('debug')
        pitch_class_four_voice, pitch_four_voice = \
            get_pitch_class_for_four_voice(sChord.recurse().getElementsByClass('Chord')[ptr], s)
        for voice_number, sonority in enumerate(pitch_four_voice):
            if sonority.isRest == False:
            ## TODO: voice number will be wrong is there is a voice crossing. This will matter when if a suspension happens here as well
            # also the voice number is inverted compared to the part number in score object
            # if sonority.pitch.midi < bass.pitch.midi:  # there is voice crossing, so we need to transpose the bass an octave lower, marked as '@'
            #     bass_lower = bass.transpose(interval.Interval(-12))
            #     aInterval = interval.Interval(noteStart=bass_lower, noteEnd=sonority)
            # else:
                aInterval = interval.Interval(noteStart=bass, noteEnd=sonority)
                colllapsed_interval = colllapse_interval(aInterval.name[1:])
                if any(colllapsed_interval in each for each in
                       fig_collapsed):  # Step 1: 7, 6, 4 can be suspensions (9 is already dealt with)
                    # Now check whether the next figure is 6, 5, 3, resepctively
                    ptr2 = 1  # this is how many onset slices we need to look ahead to get a figure
                    while sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyric in [None,
                                                                                             ' ']:  # if the there is no FB, keep searching
                        if len(sChord.recurse().getElementsByClass('Chord')) - 1 > ptr + ptr2:
                            ptr2 += 1
                        else:  # already hit the last element
                            break

                    if '7' == colllapsed_interval and any('6' in each_figure.text for each_figure in
                                                          sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyrics):

                        suspension_ptr, a_suspension = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr,
                                                          '7', a_suspension)
                    elif '6' == colllapsed_interval and any('5' in each_figure.text for each_figure in
                                                            sChord.recurse().getElementsByClass('Chord')[
                                                                ptr + ptr2].lyrics):
                        suspension_ptr, a_suspension = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr,
                                                          '6', a_suspension)
                    elif '4' == colllapsed_interval and any(each_figure.text in ['3', '#', 'b', 'n'] for each_figure in
                                                            sChord.recurse().getElementsByClass('Chord')[
                                                                ptr + ptr2].lyrics):
                        suspension_ptr, a_suspension = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr,
                                                          '4', a_suspension)
                    elif '2' ==  colllapsed_interval and any('8' in each_figure.text for each_figure in
                                                            sChord.recurse().getElementsByClass('Chord')[
                                                                ptr + ptr2].lyrics):
                        suspension_ptr, a_suspension = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr,
                                                          '2', a_suspension)
    return a_suspension




def translate_FB_into_chords(want_root_position_traid, want_suspension_NCT, want_discrepancies_chord_labels, fig, thisChord, ptr, sChord, s, number_of_space, a_suspension, a_discrepancy, a_slice_discrepancy, suspension_ptr=[]):
    """

    :param fig:
    :param thisChord:
    :return:
    """
    # if thisChord.measureNumber == 4:
    #     print('debug')
    space_needed = number_of_space - len(fig)  # align the results
    if space_needed > 0:
        for i in range(space_needed):
            thisChord.addLyric(' ')  # beautify formatting
    while len(thisChord.lyrics) > 3 and thisChord.lyrics[-1].text == ' ':  # make sure no extra space is added
        print(thisChord.lyrics)
        del thisChord.lyrics[-1]
    # the lines above is to make sure that the FBAs only take up 3 vertical lines
    chord_pitch = []
    if fig != [' '] and fig != '':
        fig_collapsed = colllapse_interval(fig)
    if '_' not in fig:  # no underline for this slice
        pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
        bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')
        if bass.name == 'rest': # no need to translate in this case
            return suspension_ptr, a_suspension
        a_suspension = suspension_processing(fig, thisChord, bass, sChord, fig_collapsed, ptr, s, a_suspension, suspension_ptr)
        # TODO: see if there is bass suspension
        # for note_number, each_note in enumerate(
        #         s.parts[-1].measure(thisChord.measureNumber).getElementsByClass(note.Note)):  # go through bass voice
        #     if each_note.beat == thisChord.beat:  # found the potential bass suspension note
        #         previous_note, previous_bass = get_previous_note(note_number, thisChord, s, -1, sChord)
        #         next_note, next_bass = get_next_note(note_number, thisChord, s, -1, sChord)
        #         if previous_note != False and next_note != False and previous_note.pitch.pitchClass == each_note.pitch.pitchClass \
        #             and (1 <= (each_note.pitch.midi - next_note.pitch.midi) <= 2) \
        #                 and sChord.recurse().getElementsByClass('Chord')[ptr + 1].orderedPitchClasses in thisChord.orderedPitchClasses:
        #                     thisChord.style.color = 'red'  # bass suspension is labelled as red
        #                     suspension_ptr.append(ptr + 1)  # TODO: cannot address bass suspension with decoration
        #                     input('bass suspension?')
                # determine bass suspension
        if fig == []:  # No figures, meaning it can have a root position triad
            thisChord.addLyric(' ') # empty line for discrepancy
            if want_root_position_traid:
                for pitch in thisChord.pitchNames:
                    chord_pitch.append(pitch)
                chord_label = chord.Chord(chord_pitch)
                allowed_chord_quality = ['major triad', 'minor triad', '-interval class 4', '-interval class 3']
                if any(each in chord_label.pitchedCommonName for each in allowed_chord_quality):
                    if bass.pitch.pitchClass == chord_label._cache['root'].pitchClass and thisChord.beat % 1 == 0:
                        if chord_label.pitchedCommonName.find('-major triad') != -1:
                            chord_name = chord_label.pitchedCommonName.replace('-major triad', '')
                        elif chord_label.pitchedCommonName.find('-minor triad') != -1:
                            chord_name = chord_label.pitchedCommonName.replace('-minor triad', 'm')
                        elif chord_label.pitchedCommonName.find('-interval class') != -1:
                            if chord_label.pitchedCommonName.find('-interval class 3') != -1:
                                chord_name = chord_label.pitchedCommonName.replace('-interval class 3', '') + 'm'
                            elif chord_label.pitchedCommonName.find('-interval class 4') != -1:
                                # four semitones apart, a major third interval
                                chord_name = chord_label.pitchedCommonName.replace('-interval class 4', '')
                            # elif chord_label.pitchedCommonName.find('-interval class 5') != -1:
                            #     chord_name = chord_label.pitchedCommonName.replace('-interval class 5', '') # missing third will be assumed as major triad, which is not perfect
                        add_chord(thisChord, chord_name)
                    else:
                        thisChord.addLyric(' ')
                else:
                    thisChord.addLyric(' ')
            else:
                thisChord.addLyric(' ')  # does not output any chord label
        else:  # there is FB
            # look at the figure bass and see which notes are included
            chord_pitch, mark = get_chord_tone(thisChord, fig, s, a_discrepancy, a_slice_discrepancy)
            chord_label = chord.Chord(chord_pitch)
            # if chord_label.pitchClasses != []:  # making sure there is no empty chord
            chord_name = is_legal_chord(chord_label)
            if chord_name:  # this slice contains a legal chord
                add_chord(thisChord, chord_name)
            else:
                if len(sChord.recurse().getElementsByClass('Chord')) > ptr + 1 and thisChord.style.color != 'pink' and \
                    want_suspension_NCT == True: # there is a next slice, but only consider it
                    # when it remains the same bass
                    # if we won't want suspension, we prefer 18th ways of labeling chord, then no replacing chords!
                    # if this is a suspension, don't replace the chord since it will be done later on anyways
                    pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
                    pitch_class_four_voice_next, pitch_four_voice_next = get_pitch_class_for_four_voice(
                        sChord.recurse().getElementsByClass('Chord')[ptr + 1], s)
                    if pitch_class_four_voice[-1] != -1 and pitch_class_four_voice_next[-1] != -1:  # both no rest
                        replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr, mark, s)
                if want_suspension_NCT == False:  # we don't want suspension, adopt the 18th way of labelling chords
                    # if the chord is not legal, output "?"
                    thisChord.addLyric('?')
                # else: # the last chord of the chorale, and it is not a legal chord
                #     add_chord(thisChord, '?' + mark)
            # consider the sonority, if there is a discrepancy
            if want_discrepancies_chord_labels:
                if '??' in mark or '?!' in mark: # in this case we need to translate based on the sonority
                    chord_label = chord.Chord(thisChord._notes)
                    # if chord_label.pitchClasses != []:  # making sure there is no empty chord
                    chord_name = is_legal_chord(chord_label)
                    if chord_name:  # this slice contains a legal chord
                        add_chord(thisChord, chord_name)

    else:  # this slice is only the continuation line, should adopt the chord from the last slice
        thisChord.addLyric(' ')  # this is the line for the discrepancy
        if any(char.isalpha() for char in sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text) \
                and 'b' not in sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text:
            # making sure it is chord label not FB, but edge case does exist (b7 maybe?)
            add_chord(thisChord, sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text)
    return suspension_ptr, a_suspension


def extract_FB_as_lyrics(path, no_instrument=False):
    # I decided to remove all the instrumental voices for now
    if not os.path.isdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi')):
        os.mkdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi'))
    f_continuation = open(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'continuation.txt'), 'w')
    for filename in os.listdir(path):
        if 'FB.musicxml' not in filename: continue
        filename_only, file_extension = os.path.splitext(filename)
        if filename_only + '_lyric.xml' in os.listdir(path): continue  # skil the files that
        # if '140.07' not in filename: continue
        # if '8.06' not in filename: continue
        print(filename, '---------------------')
        tree = ET.ElementTree(file=os.path.join(path, filename))
        parts = []
        for elem in tree.iter(tag='part'):  # get the bass voice
            parts.append(elem)
        for divisions in tree.iter(tag='divisions'):
            division = divisions.text
            break
        for beattype in tree.iter(tag='beat-type'):
            beat_type = beattype.text
            break

        child = parts[-1]  # no matter how many voices, the last one will have FB
        for measure in child.iter(tag='measure'):  # get all the measures within the bass voice
            print(measure.attrib)
            fig = []  # empty FB dictionary for each bass note
            for ele in measure.iter():
                if ele.tag == 'figured-bass' or ele.tag == 'note':
                    #print(list(ele))
                    if ele.tag == 'figured-bass':  # found a (compound) FB label
                        FB_digit = {}
                        single_digit = []  # list for all the digits
                        single_suffix = []  # list for all the suffix
                        single_prefix = []
                        extension = []
                        for each_FB_digit in ele.iter():
                            if each_FB_digit.tag == 'figure':
                                single_digit = adding_XXXfix(each_FB_digit, 'figure-number', single_digit)
                                single_suffix = adding_XXXfix(each_FB_digit, 'suffix', single_suffix)
                                single_prefix = adding_XXXfix(each_FB_digit, 'prefix', single_prefix)
                                # if each_FB_digit.find('extend') is not None:
                                #     print('debug')
                                extension = adding_XXXfix(each_FB_digit, 'extend', extension)
                                if each_FB_digit.find('extend') is not None:
                                    print(filename, file=f_continuation)
                                # print('extension is', extension)
                                FB_digit['number'] = single_digit
                                FB_digit['suffix'] = single_suffix
                                FB_digit['prefix'] = single_prefix
                                FB_digit['extension'] = extension
                            if each_FB_digit.tag == 'duration':
                                FB_digit['duration'] = str((int(each_FB_digit.text) / int(division)) * (int(beat_type) / 4))
                                # standardize dur., the first component is text/division, which will give you how many quarter note this is
                                # to get the real duration, we also need to consider the beat_type,
                                # the output is the absolute duration in beat
                        fig.append(FB_digit)
                        #print(fig)
                    if ele.tag == 'note':
                        if fig != [] and fig != [{}] and 'number':
                            #print("add the FB to lyrics")
                            # if fig[0]['number'] == ['8', '4']:
                            #     print('debug')
                            add_FB_to_lyrics(ele, fig)
                            fig = []  # reset the FB for the next note with FB
        tree.write(codecs.open(os.path.join(path, filename[:-9] + '_' + 'lyric' + '.xml'), 'w', encoding='utf-8'), encoding='unicode')
        s = converter.parse(os.path.join(path, filename[:-9] + '_' + 'lyric' + '.xml'))

        if not os.path.isdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'no_FB_as_lyrics')):
            os.mkdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'no_FB_as_lyrics'))
        s.write('midi', os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'no_FB_as_lyrics', filename + '.mid'))
        if no_instrument:
            # remove all the extra instrumental voices
            continuo_voice = contain_continuo_voice(s)
            chordify_voice = contain_chordify_voice(s)
            s = remove_instrumental_voices(s, chordify_voice, continuo_voice)
            s.write('musicxml', os.path.join(path, filename[:-9] + '_' + 'lyric_no_instrumental' + '.xml'))
    f_continuation.close()

def add_FB_align(fig, thisChord, MIDI, ptr):
    """
    Modular function to add FB in a way that the future chord labels can be aligned in one line as well
    :param fig:
    :param thisChord:
    :param MIDI: mido MIDI object to add FB as lyrics
    :return:
    """
    midi_i = -1
    midi_ptr = 0
    for j, each_message in enumerate(MIDI.tracks[-1]):
        if each_message.time != 0:
            midi_i += 1
        if midi_i == ptr:
            midi_ptr = j
            break
    FB_lyrics = " ".join(fig)
    MIDI.tracks[-1].insert(midi_ptr, MetaMessage('lyrics', text=FB_lyrics))

    for i, line in enumerate(fig):
        thisChord.addLyric(line)


def align_FB_with_slice(bassline, sChords, MIDI):
    """
    I decide to first translate all the FB as lyrics, and then translate them as lyrics, because the translation needs
    global FB to be there first.
    :param bassline:
    :param sChords:
    :param MIDI: mido MIDI object to add FB as lyrics
    :return:
    """
    flag = 0 # indicating whether the whole chorale needs to double the number since some are just half of it
    denominator_chorale = sChords.recurse().getElementsByClass(meter.TimeSignature)[0].denominator
    if denominator_chorale != 4:
        print('demoniator is', denominator_chorale, file=f_sus)
    for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
        for each_bass in bassline.measure(thisChord.measureNumber).getElementsByClass(note.Note):
            # TODO: bug found: thischord measure number does not increase if it steps into a pickup measure, the same happen for each voice, the pick up measure is not counted (included), the data of this slice is lost in music21 (except for the onset slice material)
            # TODO: if there is figure for this pick up measure, it is going to be missed
            if each_bass.beat == thisChord.beat and any(each_bass.pitch.pitchClass == each_note.pitch.pitchClass for each_note in thisChord._notes): #
                bassnote = each_bass
                if bassnote.lyrics != []:
                    fig = decode_FB_from_lyrics(bassnote.lyrics)
                    print('fig from lyrics', fig)
                    if fig[0]['number'] == ['5', '4']:
                        print('debug')
                    #print(fig)
                    displacement = 0

                    if flag == 0:
                        total_bass_duration = 0
                        for j, one_FB in enumerate(fig):
                            if 'duration' in fig[j]:
                                total_bass_duration += float(fig[j]['duration'])
                        if total_bass_duration != 0:
                            if denominator_chorale * bassnote.duration.quarterLength / 4 == total_bass_duration or len(fig) == 1: # sometimes there can be one figure over a stationary bass that has unnecessary duration, this rule should consider that too
                                flag = 0 # this is a normal piece

                            # elif denominator_chorale * bassnote.duration.quarterLength * 2 / 4 == total_bass_duration and denominator_chorale == 4:
                            #     # we need to double the duration in
                            #     # this piece as a whole
                            #     flag = 1
                            # elif denominator_chorale * bassnote.duration.quarterLength / 4 == total_bass_duration and denominator_chorale == 4:
                            #     # we need to quarple the duration in pieces like 10.07. Don't know why it is fucked up
                            #     # this piece as a whole
                            #     flag = 2
                            else:
                                print('the duration of this piece is fucked up. Look into why')

                    for j, one_FB in enumerate(fig):  # this is the place where FB should align each slice
                        slice_duration = sChords.recurse().getElementsByClass('Chord')[
                            i + j + displacement].duration.quarterLength * denominator_chorale / 4
                        if 'duration' in fig[j]:

                            # if flag == 1:
                            #     fig[j]['duration'] = str(float(fig[j]['duration']) * 2)
                            #     # don't know why some xml has half of its standard duration value
                            # elif flag == 2:
                            #     fig[j]['duration'] = str(float(fig[j]['duration']) * 4) # e.g., 10.07
                            if float(fig[j]['duration']) == slice_duration:  # this means
                                # the current FB should go to the current slice
                                add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement], MIDI, i + j + displacement)
                            else:  # the duration does not add up, meaning it should look further ahead
                                add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement], MIDI, i + j + displacement)
                                if slice_duration == total_bass_duration: # this one addresses situation with FB not found in sonority
                                    break
                                while slice_duration < float(fig[j]['duration']):
                                    displacement += 1
                                    slice_duration += sChords.recurse().getElementsByClass('Chord')[
                                        i + j + displacement].duration.quarterLength * denominator_chorale / 4
                                if slice_duration != float(fig[j]['duration']):
                                    print('duration of FB does not equal to the duration of many slices!')
                                    break

                        else:  # no duration, only one FB, just matching the current slice
                            add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement], MIDI, i + j + displacement)
                # else:
                #     thisChord.addLyric(' ')
                break


def process_suspension(sChords, want_suspension_NCT, suspension_ptr, ptr, algorithm_b_prime):
    for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
        # if i == 69:
        #     print('debug')
        # replace the suspension slices with the chord labels where it is resolved
        if want_suspension_NCT:
            if thisChord.style.color == 'pink':  # the suspensions
                look_ahead = True  # in the normal situation, we need to consider all the slices between suspension and resolution, only ignore them when the suspension already has a chord label in ALgorithm B'
                for j in range(i, suspension_ptr[ptr]):
                    if any(char.isalpha() for char in
                           sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[-1].text) \
                            and 'b' not in sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[
                        -1].text and look_ahead:
                        for label1 in sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[-2:]:
                            if label1.text[0].isalpha() and label1.text[0].isupper() and label1.text != \
                                    sChords.recurse().getElementsByClass('Chord')[j].lyrics[-2].text \
                                    and label1.text != sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text:
                                if algorithm_b_prime:
                                    if sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text == '?' or \
                                            sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text == '' or \
                                            sChords.recurse().getElementsByClass('Chord')[j].lyrics[
                                                -1].text == ' ':  # only add if the lyric is ?
                                        add_chord(sChords.recurse().getElementsByClass('Chord')[j],
                                                  label1.text)
                                    else:  # if the suspension already has a chord label, skip the whole thing
                                        look_ahead = False
                                        break
                                else:
                                    add_chord(sChords.recurse().getElementsByClass('Chord')[j],
                                              label1.text)
                ptr += 1


def lyrics_to_chordify(want_root_position_traid, want_suspension_NCT, want_discrepancies_chord_labels, path, no_instrument=False, algorithm_e=False, algorithm_b_prime = False, algorithm_de=False, translate_chord='Y'):
    a_chord_label_FB = []  # record of all the chord labels by figured bass
    a_chord_label_FB_part = [] # record chord labels by Algorithm D, later merged with the ones from Algorithm E
    a_FB = [] # record of all FB figures
    a_suspension = [] # record of all the suspensions indicated by FB
    a_discrepancy = []
    a_slice_discrepancy = []
    No_of_files = 0
    # create folder structure
    if not os.path.isdir(os.path.join(path, 'BCMCL')):
        os.mkdir(os.path.join(path, 'BCMCL'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_A')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_A'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_B')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_B'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_B_prime')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_B_prime'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_C')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_C'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_D')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_D'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_E')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_E'))
    if not os.path.isdir(os.path.join(path, 'BCMCL', 'Algorithm_DE')):
        os.mkdir(os.path.join(path, 'BCMCL', 'Algorithm_DE'))
    if algorithm_e == False and translate_chord == 'Y':
        if want_root_position_traid == False and want_suspension_NCT == False and want_discrepancies_chord_labels == False:
            f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_A_all_chords.txt'), 'w')
        elif want_root_position_traid == True and want_suspension_NCT == False and want_discrepancies_chord_labels == False:
            f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_B_all_chords.txt'), 'w')
        elif want_root_position_traid == True and want_suspension_NCT == True and want_discrepancies_chord_labels == False:
            if algorithm_b_prime:
                f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_B_prime_all_chords.txt'), 'w')
            else:
                f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_C_all_chords.txt'), 'w')
        elif want_root_position_traid == True and want_suspension_NCT == True and want_discrepancies_chord_labels == True:
            if algorithm_de == False:
                f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_D_all_chords.txt'), 'w')
            else:
                f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_DE_all_chords.txt'), 'w')
    elif translate_chord == 'Y':
        f_all_chords = open(os.path.join(path, 'BCMCL', 'Algorithm_E_all_chords.txt'), 'w')


    for filename in os.listdir(path):
        # if '83.05' not in filename or '172.06' in filename: continue
        # These lines below ensure that the algorithm only process the 120 chorale files,
        # and skip all other irrelevant ones
        if no_instrument:
            if 'lyric_no_instrumental' not in filename: continue
        else:
            if 'lyric' not in filename: continue
        if any(each_ID in filename for each_ID in ['100.06', '105.06', '113.01', '129.05', '167.05', '171.06', '24.06', '248.09', '248.23', '248.42', '248.64', '76.07' ,'8.06', '161.06a', '161.06b', '16.06', '48.07', '195.06', '149.07', '447']):
            if '124.06' not in filename and '38.06' not in filename and '168.06' not in filename and '108.06' not in filename: # don't exclude this one!
                continue  # exclude all the interlude chorales
        # if filename[:-4] + '_chordify' + filename[-4:] in os.listdir(path) and translate_chord == 'Y':
        #     continue  # don't need to translate the chord labels if already there
        if 'chordify' in filename: continue
        if '2_voice'  in filename: continue
        if 'FB_align' in filename: continue
        # if '33.06' not in filename: continue
        # if '133.06' in filename: continue
        No_of_files += 1
        # if No_of_files > 5: continue
        print(No_of_files)
        print(filename)
        if translate_chord == 'Y':
            print(filename, file=f_all_chords)
            print(filename, file=f_sus)
        suspension_ptr = []  # list that records all the suspensions
        ptr = 0  # record how many suspensions we have within this chorale
        s_MIDI = converter.parse(os.path.join(path,
                                              filename))  # output a MIDI file with onset slices to add FB as lyric meta messages
        s_MIDI_Chords = s_MIDI.chordify()
        s_MIDI.insert(0, s_MIDI_Chords)
        if not os.path.isdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'with_FB_as_lyrics')):
            os.mkdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'with_FB_as_lyrics'))
        s_MIDI.write('midi', os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'with_FB_as_lyrics',
                                          filename + '.mid'))
        s = converter.parse(os.path.join(path, filename))
        # TODO: figure out why the MIDI version still transpose bass an octave lower
        MIDI = MidiFile(os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'translated_midi', 'with_FB_as_lyrics',
                                     filename + '.mid'))
        for n in s.parts[-1].recurse().notes:
            n.transpose(interval.Interval('P-8'), inPlace=True)  # don't use -12, since the spelling is messed up!
        # transpose bass down an octave to avoid voice crossings
        bassline = s.parts[-1]
        sChords = s.chordify()
        align_FB_with_slice(bassline, sChords, MIDI)
        if translate_chord == 'Y':
            if algorithm_e == False :
                for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    if i == 42:
                        print('debug')
                    fig = get_FB(sChords, i)
                    if fig != []:
                        print(fig)
                    if fig == ['5', '4', '2']:
                        print('debug')
                    #     print(thisChord.lyrics)
                    a_FB.append(fig)
                    suspension_ptr, a_suspension = translate_FB_into_chords(want_root_position_traid, want_suspension_NCT,
                                                                 want_discrepancies_chord_labels,
                                                                 fig, thisChord, i, sChords, s, 3, a_suspension,
                                                                 a_discrepancy,
                                                                 a_slice_discrepancy, suspension_ptr)
                    # the line above is the entry of the main chord translation function
                    thisChord.closedPosition(forceOctave=4, inPlace=True)  # if you put it too early, some notes including an
                    # octave apart will be collapsed!
                if algorithm_de == True:
                    process_suspension(sChords, want_suspension_NCT, suspension_ptr, 0, algorithm_b_prime)
                    s.insert(0, sChords)
                    suspension_ptr = []
            if algorithm_de == True or algorithm_e ==True: # Algorithm E which translates chord label based on sonority only.
                k = s.analyze('AardenEssen')
                IR = s.chordify()
                for j, c in enumerate(IR.recurse().getElementsByClass('Chord')):
                    c.closedPosition(forceOctave=4, inPlace=True)
                    pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(c, s)
                    bass = get_bass_note(c, pitch_four_voice, pitch_class_four_voice, 'Y')
                    if bass.name == 'rest': continue
                    intervals = []  # store all the exhaustive FB
                    for sonority in pitch_four_voice:
                        if hasattr(sonority, 'pitch') and bass.name != 'rest':
                            intervals = get_actual_figures(bass, sonority, intervals, k)

                    for i, line in enumerate(intervals):
                        if line == '8': continue
                        c.addLyric(line)
                    # c.annotateIntervals()
                 # attach the chord label results from Algorithm D before it disappears
                sChords = s.chordify()  # this is used because annotateIntervals only works properly if I collapse octaves, but for suspension, I need the uncollapsed version,
                # so I need to create a new chordify voice
                for j, c in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    for each_line in IR.recurse().getElementsByClass('Chord')[j].lyrics:
                        # add_chord(c, each_line.text)  # copy lyrics to this voice
                        c.addLyric(each_line.text)
                for j, c in enumerate(
                        sChords.recurse().getElementsByClass('Chord')):  # identifying suspension needs full figures
                    fig = get_FB(sChords, j)
                    # if fig != []:
                    #     print(fig)
                    for k, figure_unicode in enumerate(fig):  # replace all the unicode with #, b, and n
                        fig[k] = fig[k].replace('♯', '#')
                        fig[k] = fig[k].replace('♭', 'b')
                        fig[k] = fig[k].replace('♮', 'n')
                    # if fig == ['4', '5', '4']:
                    #     print('debug')
                    # a_IR.append(fig)
                    if algorithm_de == True:
                        a_suspension_fake = list(a_suspension)
                        suspension_ptr, _ = translate_FB_into_chords(want_root_position_traid,
                                                                                want_suspension_NCT,
                                                                                want_discrepancies_chord_labels,
                                                                                fig, c, j, sChords, s, 4, a_suspension_fake,
                                                                                a_discrepancy, a_slice_discrepancy,
                                                                                suspension_ptr)  # I don't need to calculate suspension in IR voice!
                    else:
                        suspension_ptr, a_suspension = translate_FB_into_chords(want_root_position_traid, want_suspension_NCT,
                                             want_discrepancies_chord_labels,
                                             fig, c, j, sChords, s, 4, a_suspension, a_discrepancy, a_slice_discrepancy,
                                             suspension_ptr)  # I don't need to calculate suspension in IR voice!
                    c.closedPosition(forceOctave=4, inPlace=True)
                    c.lyrics[-1].text = c.lyrics[-1].text.replace('?!',
                                                                  '')  # TODO: look into why this happens later! Some root position chords has this unnecessary ?! sign
            process_suspension(sChords, want_suspension_NCT, suspension_ptr, 0, algorithm_b_prime)
        else:
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                thisChord.closedPosition(forceOctave=4, inPlace=True)
        s.insert(0, sChords)  # attach the chord label results
        if translate_chord == 'Y': # write chord results to files
            if algorithm_e == False:
                if want_root_position_traid == False and want_suspension_NCT == False and want_discrepancies_chord_labels == False:
                    s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_A', filename[:-4] + '_' + 'chordify_algorithm_A' + '.xml'))
                elif want_root_position_traid == True and want_suspension_NCT == False and want_discrepancies_chord_labels == False:
                    s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_B', filename[:-4] + '_' + 'chordify_algorithm_B' + '.xml'))
                elif want_root_position_traid == True and want_suspension_NCT == True and want_discrepancies_chord_labels == False:
                    if algorithm_b_prime:
                        s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_B_prime',
                                                         filename[:-4] + '_' + 'chordify_algorithm_B_prime' + '.xml'))
                    else:
                        s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_C', filename[:-4] + '_' + 'chordify_algorithm_C' + '.xml'))
                elif want_root_position_traid == True and want_suspension_NCT == True and want_discrepancies_chord_labels == True:
                    if algorithm_de == False:
                        s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_D', filename[:-4] + '_' + 'chordify_algorithm_D' + '.xml'))
                    else:
                        s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_DE',
                                                         filename[:-4] + '_' + 'chordify_algorithm_DE' + '.xml'))
            else:
                s.write('musicxml', os.path.join(path, 'BCMCL', 'Algorithm_E',
                                                 filename[:-4] + '_' + 'chordify_algorithm_E' + '.xml'))
        else:
            s.write('musicxml', os.path.join(path,
                                             filename[:-4] + '_' + 'FB_align' + '.xml'))
        # obtain chord labels and do some statistical analysis
        if translate_chord == 'Y':
            voice_FB = s.parts[-1]
            a_chord_label_FB, all_chord_for_this_file = put_chords_into_files(voice_FB, a_chord_label_FB, replace='N')
            if algorithm_de == True:
                voice_FB_algorithm_D = s.parts[-2]
                a_chord_label_FB_part_final = []
                a_chord_label_FB_part, all_chord_for_this_file_part = put_chords_into_files(voice_FB_algorithm_D, a_chord_label_FB_part, replace='N')
                # merge together
                for id, each_chord in enumerate(a_chord_label_FB_part):
                    a_chord_label_FB_part_final.append(list(set(each_chord).union(a_chord_label_FB[id])))
                print('debug')
                a_chord_label_FB = all_chord_for_this_file = a_chord_label_FB_part_final
            print(all_chord_for_this_file, file=f_all_chords)
    if algorithm_e == False and translate_chord == 'Y':
        a_chord_label_final_only_multiple_interpretations = []
        a_chord_label_final_only_multiple_interpretations_key_invariant = []
        for each in a_chord_label_FB:
            if len(each) > 1:
                a_chord_label_final_only_multiple_interpretations.append(','.join(each))
                a_chord_label_final_only_multiple_interpretations_key_invariant.append(key_invariant_pairs(each))
        a_chord_label_FB_flat = list(itertools.chain.from_iterable(a_chord_label_FB))
        # get chord quality
        a_chord_quality = []
        for each_chord in a_chord_label_FB_flat:
            if '#' in each_chord or '-' in each_chord:
                a_chord_quality.append(each_chord[2:])
            else:
                a_chord_quality.append(each_chord[1:])
        print_distribution_plot('Multiple Interpretations', a_chord_label_final_only_multiple_interpretations,a_chord_label_FB)
        print_distribution_plot('Labels Both Valid for a Chord', a_chord_label_final_only_multiple_interpretations_key_invariant,
                                a_chord_label_FB)
        print_distribution_plot('Chord Types', a_chord_label_FB_flat, a_chord_label_FB)
        print_distribution_plot('Chord Qualities', a_chord_quality, a_chord_label_FB)
        print_distribution_plot('Suspensions', a_suspension, a_chord_label_FB)
        print_distribution_plot('Discrepancies Between Figures and Surface', a_discrepancy, a_chord_label_FB, a_slice_discrepancy)
        print('there are altogether', len(a_chord_label_FB), 'onset slices and there are', len(a_chord_label_FB_flat), 'chord labels')
        # print('debug')
        f_all_chords.close()

def print_distribution_plot(word, unit, total_NO_slice, a_slice_discrepancy=[]):
    if word == 'Multiple Interpretations':  # in this case, we want to collapse "D, D7" and "D7, D" into one category
        # print('debug')
        for i, each_item in enumerate(unit):
            elements = each_item.split(',')
            for ii, each_chord in enumerate(elements):
                elements[ii] = elements[ii].replace(' ', '')
            unit[i] = ','.join(sorted(elements))
    unit_dict = Counter(unit)
    if word == 'Suspensions':
        unit_dict['4–3'] = unit_dict['4']
        del unit_dict['4']
        unit_dict['7–6'] = unit_dict['7']
        del unit_dict['7']
        unit_dict['9–8'] = unit_dict['2']
        del unit_dict['2']
        unit_dict['6–5'] = unit_dict['6']
        del unit_dict['6']
        unit_dict['64–53'] = unit_dict['64']
        del unit_dict['64']
    elif word == 'Chord Qualities':
        unit_dict['M'] = unit_dict['']
        del unit_dict['']
    top_N = 20
    if word != 'Discrepancies Between Figures and Surface':
        print('there is', sum(unit_dict.values()), sum(unit_dict.values())/len(total_NO_slice) * 100, '%', word)
    else:
        slice_with_discrepancies = len(a_slice_discrepancy)
        print('there is', slice_with_discrepancies, slice_with_discrepancies / len(total_NO_slice) * 100, '%', word)
    counter = dict(unit_dict.most_common())# sort the dic
    counter_fre = turn_number_into_percentage(counter) # I want percentage of each class
    counter_fre_top_N = take_top_N(counter_fre, top_N)

    print('there are', len(counter_fre), word, 'and the distribution of them is:', counter_fre_top_N, 'and the full one is:', counter_fre)
    if word == 'Multiple Interpretations':
        num_of_three_interpretation = 0
        num_of_four_interpretation = 0
        num_of_two_interpretation = 0
        num_of_five_interpretation = 0
        for each_label in unit:
            multiple = each_label.split(',')
            if len(multiple) == 3:
                num_of_three_interpretation += 1
            elif len(multiple) == 4:
                num_of_four_interpretation += 1
            elif len(multiple) == 2:
                num_of_two_interpretation += 1
            elif len(multiple) == 5:
                num_of_five_interpretation += 1
        print('there are', num_of_four_interpretation, 'four interpretations',
              num_of_four_interpretation / len(total_NO_slice) * 100, '%')
        print('there are', num_of_three_interpretation, 'three interpretations', num_of_three_interpretation/len(total_NO_slice) * 100, '%')
        print('there are', num_of_two_interpretation, 'two interpretations',
              num_of_two_interpretation / len(total_NO_slice) * 100, '%')
        print('there are', num_of_five_interpretation, 'five interpretations',
              num_of_five_interpretation / len(total_NO_slice) * 100, '%')
    plt.bar(list(counter_fre_top_N.keys()), counter_fre_top_N.values(), width=1, color='g')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Percentage (%)')
    plt.xlabel(word)
    plt.xticks(rotation='vertical')
    #plt.show()

def turn_number_into_percentage(c):
    cc = {}
    s = sum(c.values())
    for elem, count in c.items():
        cc[elem] = count / s
    return cc


def take_top_N(dictionary, num):
    """
    Only output the first N categories, and for the rest it is collapsed into "other"
    :param dict:
    :return:
    """
    top_dict = dict(itertools.islice(dictionary.items(),num))
    other_frequency = sum(dict(list(dictionary.items())[num:]).values())
    if len(dictionary) > num:
        top_dict['other'] = other_frequency
    return top_dict


if __name__ == '__main__':
    path = os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'musicXML_master')
    no_instrument = False
    # Step 1: we need to first extract the figured bass anntoations from BCFB into a format music21 can process
    extract_FB_as_lyrics(path, no_instrument)
    lyrics_to_chordify(False, False, False, path, no_instrument, translate_chord='N')
    # Step 2: choose an algorithm you want to use to generate chord labels
    # For BCMCL 1.1, you don't need to change the script
    # lyrics_to_chordify(False, False, False, path, no_instrument) # Algorithm A
    # lyrics_to_chordify(True, False, False, path, no_instrument) # Algorithm B
    lyrics_to_chordify(True, True, False, path, no_instrument, algorithm_e=False, algorithm_b_prime=True) # Algorithm B'
    lyrics_to_chordify(True, True, False, path, no_instrument) # Algorithm C
    lyrics_to_chordify(True, True, True, path, no_instrument) # Algorithm D
    lyrics_to_chordify(False, True, False, path, no_instrument, algorithm_e=True) # Algorithm E
    # lyrics_to_chordify(True, True, True, path, no_instrument, algorithm_de=True)
    # This is to calculate the statistics of BCMCL 1.1 reported in Section

