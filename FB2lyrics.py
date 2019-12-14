import xml.etree.cElementTree as ET
import os
from music21 import *
import re
from get_input_and_output import get_pitch_class_for_four_voice


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
                             'augmented triad']
    chord_name = chord_label.pitchedCommonName
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
                chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[
                                                            :harmony.chordSymbolFigureFromChord(chord_label).find(
                                                                'add')])  # remove 'add' part
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
        return chord_name
    else:
        return None


def colllapse_interval(string):
    """

    :param str:
    :return:
    """
    if type(string) == str:
        if int(string) % 7 == 0:
            return '7'
        else:
            return str(int(string) % 7)  # collapse all the intervals within an octave
    elif type(string) == list:
        string_2 = list(string)  # preserve the original FB
        for i, each_figure in enumerate(string_2):
            if each_figure.find('9') != -1:
                string_2[i] = '2'  #  9 as 2,
            if each_figure.find('8') != -1:
                string_2[i] = '1'  #  9 as 2,
        return string_2


def get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, note='N'):
    """
    Little function deciding whether the bass note should be from the former or the latter
    :param thisChord:
    :param pitch_four_voice:
    :return:
    """
    if pitch_class_four_voice[-1] != -1:  # if bass is not rest:
        if note == 'Y':
            bass = pitch_four_voice[-1]  # I want a note object, since the interval.interval function requires a note
        else:
            bass = pitch_four_voice[-1].pitch  # default
    elif pitch_class_four_voice[-2] != -1:  # if bass is rest, which is pretty rare, then get whatever it is in thisChord
        bass = pitch_four_voice[-2]  # in this case, the bass is tenor
    elif pitch_class_four_voice[-3] != -1:
        bass = pitch_four_voice[-3]  # in this case, the bass is alto
    else:
        bass = pitch_four_voice[-4]  # in this case, the bass is soprano
    return bass

def get_chord_tone(thisChord, fig, s, condition='N'):
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
    #print('current fig is', fig)
    if fig != [' '] and fig != '':
        fig_collapsed = colllapse_interval(fig)
    if condition == 'N': # this means we need to choose which ones are CT based on the FB
        mark = ''  # indicating weird slices
        intervals = []
        pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
        bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')
        for i, note in enumerate(thisChord._notes):
            # print(note)
            # print('bass', bass)
            if note.pitch.midi < bass.pitch.midi: # there is voice crossing, so we need to transpose the bass an octave lower, marked as '@'
                mark += '@'
                bass_lower = bass.transpose(interval.Interval(-12))
                aInterval = interval.Interval(noteStart=bass_lower, noteEnd=note)
            else:
                aInterval = interval.Interval(noteStart=bass, noteEnd=note)
            colllapsed_interval = colllapse_interval(aInterval.name[1:])
            intervals.append(colllapsed_interval)
            # TODO: there might be cases where we need to check whether there is a real 9, or just a 2. In this case we cannot check
            if ('3' in colllapsed_interval and '4' not in fig_collapsed and '2' not in fig_collapsed) or ('5' in colllapsed_interval and '6' not in fig_collapsed) or '1' in colllapsed_interval:
                chord_pitch.append(note)
            elif any(colllapsed_interval in each for each in fig_collapsed):
                chord_pitch.append(note)
            elif colllapsed_interval == '6' and (fig_collapsed == ['4', '3'] or fig_collapsed == ['4', '2'] or fig_collapsed == ['2']):  # TODO: check if there is 246 case happen
                chord_pitch.append(note)
            elif colllapsed_interval == '4' and fig_collapsed == ['2']:
                chord_pitch.append(note)
            else: # sonority not in the FB
                mark += '??'
        for each_figure in fig_collapsed:
            if each_figure == '' or '_' in each_figure:
                continue
            if each_figure[-1] not in intervals:  # FB not in sonorities
                if each_figure in ['n', '#', 'b'] and '3' in intervals: # this is an exception
                    mark += ''
                else:
                    mark += '?!'
        return chord_pitch, mark
    else:
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
    if chordname[-1] == '-':  # This is the bug music21 has: for chords with flats, it will fail to add as lyrics
        thisChord.addLyric(chordname + '*')  # adding a casual sign (then remove it) to avoid this bug
        thisChord.lyrics[-1].text = thisChord.lyrics[-1].text.replace('*', '')
    else:
        thisChord.addLyric(chordname)


def label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr, sus_type):
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
    if is_suspension(ptr, ptr2, s, sChord, 3 - voice_number, sus_type):
        thisChord.style.color = 'pink'
        suspension_ptr.append(ptr + ptr2)
    return suspension_ptr


def get_previous_note(note_number, thisChord, s, voice_number):
    """
    Modular function to get the previous note for suspension
    :return:
    """
    if note_number == 0:  # this means the previous note is in last measure:
        if thisChord.measureNumber > 1:  # Edge cases: IR can have this edge case
            previous_note = \
                s.parts[voice_number].measure(thisChord.measureNumber - 1).getElementsByClass(note.Note)[-1]
        else:
            return False  # this edge case where there is not even a previous note, let alone will be susupension
    else:
        previous_note = s.parts[voice_number].measure(thisChord.measureNumber).getElementsByClass(note.Note)[
            note_number - 1]
    return previous_note


def get_next_note(note_number, thisChord, s, voice_number):
    """
    Modular function to get the next note for suspension
    :return:
    """
    if note_number == len(s.parts[voice_number].measure(thisChord.measureNumber).getElementsByClass(note.Note)) - 1:  # this means the next note is in next measure:
        if s.parts[voice_number].measure(thisChord.measureNumber + 1) is not None:  # It has the next measure
            next_note = \
                s.parts[voice_number].measure(thisChord.measureNumber + 1).getElementsByClass(note.Note)[0]
        else:
            return False  # this edge case where there is not even a previous note, let alone will be susupension
    else:
        next_note = s.parts[voice_number].measure(thisChord.measureNumber).getElementsByClass(note.Note)[
            note_number + 1]
    return next_note


def is_suspension(ptr, ptr2, s, sChord, voice_number, sus_type):
    """
    For possible suspension figures (e.g., 7+6, 6+5, 4+3), test if contrapuntally speaking it is a suspension or not
    :return:
    """
    thisChord = sChord.recurse().getElementsByClass('Chord')[ptr]
    pitch_class_four_voice, pitch_four_voice = \
        get_pitch_class_for_four_voice(thisChord, s)
    ## find which voice does this note live
    # for real_voice_number, each_note in enumerate(pitch_four_voice) :
    #     if each_note == thisChord._notes[voice_number]:
    #         if real_voice_number != voice_number:  # There can be two edge cases: (1) voice crossing and (2) two voices
    #             #share the same note
    #             input('how to deal with these two edge cases?')
    # TODO: use this section of code above to find the edge cases

    pitch_class_four_voice_next, pitch_four_voice_next = get_pitch_class_for_four_voice(
        sChord.recurse().getElementsByClass('Chord')[ptr +  ptr2], s)
    if pitch_class_four_voice[-1] != -1 and pitch_class_four_voice_next[-1] != -1:  # both no rest
        if pitch_four_voice[-1].pitch.pitchClass == pitch_four_voice_next[-1].pitch.pitchClass:  # bass remains the same or same pitch class coz sometimes there can be a decoration in between (e.g., 050 last measure), (1)
            for note_number, each_note in enumerate(s.parts[voice_number].measure(thisChord.measureNumber).getElementsByClass(note.Note)):
                if each_note.beat == thisChord.beat: # found the potential suspension note
                    previous_note = get_previous_note(note_number, thisChord, s, voice_number)
                    if previous_note == False:
                        return False
                    next_note = get_next_note(note_number, thisChord, s, voice_number)
                    if next_note == False:
                        return False
                    if previous_note.pitch.pitchClass == each_note.pitch.pitchClass and (1 <= (each_note.pitch.midi - next_note.pitch.midi) <= 2 or sus_type == '6'):  # the previous note and the current note should be the same, or in the same pitch class (2)
                        # and also the note should resolve downstep (3), or it is a 6-5 suspension
                        return True
                elif each_note.beat < thisChord.beat and (each_note.beat + each_note.duration.quarterLength > thisChord.beat): # It is possible that the "previous" note sustains through the suspended slice
                    next_note = get_next_note(note_number, thisChord, s, voice_number)
                    if next_note == False:
                        return False
                    if 1 <= (each_note.pitch.midi - next_note.pitch.midi) <= 2:
                        return True
    return False


def replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr, mark, s):
    if pitch_four_voice[-1].pitch.pitchClass == pitch_four_voice_next[-1].pitch.pitchClass or int(
            thisChord.beat) == int(sChord.recurse().getElementsByClass('Chord')[ptr + 1].beat):
        # same bass or different basses but same beat (362 mm.2 last)
        next_chord_pitch, next_mark = get_chord_tone(sChord.recurse().getElementsByClass('Chord')[ptr + 1], '', s,
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
                sChord.recurse().getElementsByClass('Chord')[ptr + 1], '', s,
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



def translate_FB_into_chords(fig, thisChord, ptr, sChord, s, suspension_ptr=[]):
    """

    :param fig:
    :param thisChord:
    :return:
    """
    chord_pitch = []
    if fig != [' '] and fig != '':
        fig_collapsed = colllapse_interval(fig)
    if fig != ['_']:  # no underline for this slice
        pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
        bass = get_bass_note(thisChord, pitch_four_voice, pitch_class_four_voice, 'Y')

        if '7' in fig or '6' in fig or '4' in fig:  # In these cases, examine whether this note is a suspension or not
            # if thisChord.measureNumber == 12:
            #     print('debug')
            for voice_number, note in enumerate(thisChord._notes):
                ## TODO: voice number will be wrong is there is a voice crossing. This will matter when if a suspension happens here as well
                # also the voice number is inverted compared to the part number in score object
                if note.pitch.midi < bass.pitch.midi: # there is voice crossing, so we need to transpose the bass an octave lower, marked as '@'
                    bass_lower = bass.transpose(interval.Interval(-12))
                    aInterval = interval.Interval(noteStart=bass_lower, noteEnd=note)
                else:
                    aInterval = interval.Interval(noteStart=bass, noteEnd=note)
                colllapsed_interval = colllapse_interval(aInterval.name[1:])
                if any(colllapsed_interval in each for each in fig_collapsed):  # Step 1: 7, 6, 4 can be suspensions (9 is already dealt with)
                    # Now check whether the next figure is 6, 5, 3, resepctively
                    ptr2 = 1  # this is how many onset slices we need to look ahead to get a figure
                    while sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyric in [None, ' ']:  # if the there is no FB, keep searching
                        if len(sChord.recurse().getElementsByClass('Chord')) - 1 > ptr + ptr2:
                            ptr2 += 1
                        else:  # already hit the last element
                            break

                    if '7' == colllapsed_interval and any('6' in each_figure.text for each_figure in sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyrics):

                        suspension_ptr = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr, '7')
                    elif '6' == colllapsed_interval and any('5' in each_figure.text for each_figure in sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyrics):
                        suspension_ptr = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr, '6')
                    elif '4' == colllapsed_interval and any(each_figure.text in ['3', '#', 'b', 'n'] for each_figure in sChord.recurse().getElementsByClass('Chord')[ptr + ptr2].lyrics):
                        suspension_ptr = label_suspension(ptr, ptr2, s, sChord, voice_number, thisChord, suspension_ptr, '4')
                    # possible_suspension = note
                    # thisChord.addLyric('SUS')

            # if is_suspension():
            #     # highlight the note
        if fig == []:  # No figures, meaning it can have a root position triad
            for pitch in thisChord.pitchNames:
                chord_pitch.append(pitch)
            chord_label = chord.Chord(chord_pitch)
            allowed_chord_quality = ['major triad', 'minor triad']
            if any(each in chord_label.pitchedCommonName for each in allowed_chord_quality):
                if bass.pitch.pitchClass == chord_label._cache['root'].pitchClass and thisChord.beat % 1 == 0:
                    if chord_label.pitchedCommonName.find('-major triad') != -1:
                        chord_name = chord_label.pitchedCommonName.replace('-major triad', '')
                    else:
                        chord_name = chord_label.pitchedCommonName.replace('-minor triad', 'm')
                    add_chord(thisChord, chord_name)
                else:
                    thisChord.addLyric(' ')
            else:
                thisChord.addLyric(' ')
        else:  # there is FB
            # look at the figure bass and see which notes are included
            chord_pitch, mark = get_chord_tone(thisChord, fig, s)
            chord_label = chord.Chord(chord_pitch)
            chord_name = is_legal_chord(chord_label)
            if chord_name:  # this slice contains a legal chord
                add_chord(thisChord, mark + chord_name)
            else:


                if len(sChord.recurse().getElementsByClass('Chord')) > ptr + 1: # there is a next slice, but only consider it
                    # when it remains the same bass
                    pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
                    pitch_class_four_voice_next, pitch_four_voice_next = get_pitch_class_for_four_voice(
                        sChord.recurse().getElementsByClass('Chord')[ptr + 1], s)
                    if pitch_class_four_voice[-1] != -1 and pitch_class_four_voice_next[-1] != -1:  # both no rest
                        replace_with_next_chord(pitch_four_voice, pitch_four_voice_next, thisChord, sChord, ptr, mark, s)
                else: # the last chord of the chorale, and it is not a legal chord
                    add_chord(thisChord, '?' + mark)
    else:  # this slice is only the continuation line, should adopt the chord from the last slice
        if any(char.isalpha() for char in sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text) \
                and 'b' not in sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text:
            # making sure it is chord label not FB, but edge case does exist (b7 maybe?)
            add_chord(thisChord, sChord.recurse().getElementsByClass('Chord')[ptr - 1].lyrics[-1].text)
    return suspension_ptr


def extract_FB_as_lyrics():
    for filename in os.listdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source')):
        if 'FB.musicxml' not in filename: continue
        # if '013' not in filename: continue
        print(filename, '---------------------')
        tree = ET.ElementTree(file=os.path.join('.', 'Bach_chorale_FB', 'FB_source', filename))
        for elem in tree.iter(tag='part'):  # get the bass voice
            if elem.attrib['id'] == 'P4':
                child = elem
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
                                print('extension is', extension)
                                FB_digit['number'] = single_digit
                                FB_digit['suffix'] = single_suffix
                                FB_digit['prefix'] = single_prefix
                                FB_digit['extension'] = extension
                            if each_FB_digit.tag == 'duration':
                                FB_digit['duration'] = each_FB_digit.text
                        fig.append(FB_digit)
                        print(fig)
                    if ele.tag == 'note':
                        if fig != [] and fig != [{}] and 'number':
                            print("add the FB to lyrics")
                            # if fig[0]['number'] == ['8', '4']:
                            #     print('debug')
                            add_FB_to_lyrics(ele, fig)
                            fig = []  # reset the FB for the next note with FB
        tree.write(open(os.path.join('.', 'Bach_chorale_FB', 'FB_source', filename[:-9] + '_' + 'lyric' + '.xml'), 'w'), encoding='unicode')


def add_FB_align(fig, thisChord):
    """
    Modular function to add FB in a way that the future chord labels can be aligned in one line as well
    :param fig:
    :param thisChord:
    :param ptr:
    :return:
    """
    for i, line in enumerate(fig):
        thisChord.addLyric(line)
    # put space for future chord labels to align
    # if len(fig) == 1:
    #     thisChord.addLyric(' ')
    #     thisChord.addLyric(' ')
    # elif len(fig) == 2:
    #     thisChord.addLyric(' ')


def align_FB_with_slice(bassline, sChords):
    """
    I decide to first translate all the FB as lyrics, and then translate them as lyrics, because the translation needs
    global FB to be there first.
    :param bassline:
    :param sChords:
    :return:
    """
    for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
        for each_bass in bassline.measure(thisChord.measureNumber).getElementsByClass(note.Note):
            if each_bass.beat == thisChord.beat:
                bassnote = each_bass
                if bassnote.lyrics != []:
                    fig = decode_FB_from_lyrics(bassnote.lyrics)
                    #print(fig)
                    displacement = 0
                    denominator_chorale = sChords.recurse().getElementsByClass(meter.TimeSignature)[0].denominator
                    for j, one_FB in enumerate(fig):  # this is the place where FB should align each slice
                        slice_duration = sChords.recurse().getElementsByClass('Chord')[
                            i + j + displacement].duration.quarterLength
                        if 'duration' in fig[j]:
                            if slice_duration * 2 == float(fig[j]['duration']) and denominator_chorale == 4:
                                fig[j]['duration'] = str(float(fig[j]['duration']) * 2)
                                # don't know why some xml has half of its standard duration value
                            if float(fig[j]['duration']) / float(denominator_chorale) == slice_duration:  # this means
                                # the current FB should go to the current slice
                                add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement])
                            else:  # the duration does not add up, meaning it should look further ahead
                                add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement])
                                while slice_duration < float(fig[j]['duration']) / float(denominator_chorale):
                                    displacement += 1
                                    slice_duration += sChords.recurse().getElementsByClass('Chord')[
                                        i + j + displacement].duration.quarterLength
                                if slice_duration != float(fig[j]['duration']) / float(denominator_chorale):
                                    print('duration of FB does not equal to the duration of many slices!')

                        else:  # no duration, only one FB, just matching the current slice
                            add_FB_align(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j + displacement])
                else:
                    thisChord.addLyric(' ')
                break


def get_FB(sChords, ptr):
    """
    Get FB from lyrics with multiple lines (indicating multiple figures)
    :return:
    """
    fig = []
    for each_line in sChords.recurse().getElementsByClass('Chord')[ptr].lyrics:
        fig.append(each_line.text)
    if fig == [' ']:
        fig = []
    return fig

def lyrics_to_chordify(want_IR):
    for filename in os.listdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source')):
        if 'lyric' not in filename: continue
        if filename[:-4] + '_chordify' + filename[-4:] in os.listdir(os.path.join('.', 'Bach_chorale_FB', 'FB_source')):
            continue  # don't need to translate the chord labels if already there
        if 'chordify' in filename: continue
        if '043' not in filename: continue
        print(filename)
        suspension_ptr = []  # list that records all the suspensions
        ptr = 0  # record how many suspensions we have within this chorale
        s = converter.parse(os.path.join('.', 'Bach_chorale_FB', 'FB_source', filename))
        for n in s.parts[-1].recurse().notes:
            n.transpose(interval.Interval('P-8'), inPlace=True)  # don't use -12, since the spelling is messed up!
        # transpose bass down an octave to avoid voice crossings
        bassline = s.parts[-1]
        sChords = s.chordify()
        align_FB_with_slice(bassline, sChords)
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
            fig = get_FB(sChords, i)
            if fig != []:
                print(fig)
            if fig == ['6', '3']:
                print('debug')
            suspension_ptr = translate_FB_into_chords(fig, thisChord, i, sChords, s, suspension_ptr)
            thisChord.closedPosition(forceOctave=4, inPlace=True)  # if you put it too early, some notes including an
            # octave apart will be collapsed!
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
        # replace the suspension slices with the chord labels where it is resolved
            if thisChord.style.color == 'pink':  # the suspensions
                for j in range(i, suspension_ptr[ptr]):
                    if any(char.isalpha() for char in
                        sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[-1].text) \
                    and 'b' not in sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[-1].text:
                        sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text\
                            = sChords.recurse().getElementsByClass('Chord')[suspension_ptr[ptr]].lyrics[-1].text
                        sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text = \
                            sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text.replace('?', '')
                        sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text = \
                            sChords.recurse().getElementsByClass('Chord')[j].lyrics[-1].text.replace('!', '')
                ptr += 1

        if want_IR:
            IR = s.chordify()
            for j, c in enumerate(IR.recurse().getElementsByClass('Chord')):
                c.closedPosition(forceOctave=4, inPlace=True)
                c.annotateIntervals()
            IR2 = s.chordify()  # this is used because annotateIntervals only works properly if I collapse octaves, but for suspension, I need the uncollapsed version,
            # so I need to create a new chordify voice
            for j, c in enumerate(IR2.recurse().getElementsByClass('Chord')):
                for each_line in IR.recurse().getElementsByClass('Chord')[j].lyrics:
                    add_chord(c, each_line.text)  # copy lyrics to this voice
            for j, c in enumerate(IR2.recurse().getElementsByClass('Chord')):  # identifying suspension needs full figures
                fig = get_FB(IR2, j)
                if fig != []:
                    print(fig)
                if fig == ['6', '3']:
                    print('debug')
                translate_FB_into_chords(fig, c, j, IR2, s)
                c.closedPosition(forceOctave=4, inPlace=True)
            s.insert(0, sChords)
            s.insert(0, IR2)
        else:
            s.insert(0, sChords)
        s.write('musicxml', os.path.join('.', 'Bach_chorale_FB', 'FB_source', filename[:-4] + '_' + 'chordify' + '.xml'))


if __name__ == '__main__':
    want_IR = True
    # extract_FB_as_lyrics()
        # till this point, all FB has been extracted and attached as lyrics underneath the bass line!
    lyrics_to_chordify(want_IR)

