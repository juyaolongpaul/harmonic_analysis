import xml.etree.cElementTree as ET
import os
from music21 import *
import re


def translate_FB_as_lyrics(number, suffix, prefix):
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
    return prefix_sign+suffix_sign+number


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
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i]) + '+' + each_fb['duration']
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i]) + '+' + each_fb['duration']
        else:
            if text.text is None:
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i])
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i])
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
        print('numer of layers', i)
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
            if '4' in each_figure_duration:
                print('debug')
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


def translate_FB_into_chords(fig, thisChord, ptr):
    """

    :param fig:
    :param thisChord:
    :return:
    """
    chord_pitch = []
    if fig == '':  # No figures, meaning it can have a root position triad
        for pitch in thisChord.pitchNames:
            chord_pitch.append(pitch)
        chord_label = chord.Chord(chord_pitch)
        allowed_chord_quality = [ 'major triad', 'minor triad']
        if any(each in chord_label.pitchedCommonName for each in allowed_chord_quality):
            if thisChord.bass().pitchClass == chord_label._cache['root'].pitchClass and thisChord.beat % 1 == 0:  # TODO: thisChord.bass() might not be correct to get the bass note!
                if chord_label.pitchedCommonName.find('-major triad') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-major triad', '')
                else:
                    chord_name = chord_label.pitchedCommonName.replace('-minor triad', 'm')
                thisChord.addLyric(chord_name)
            else:
                thisChord.addLyric(' ')
        else:
            thisChord.addLyric(' ')
    else:  # there is FB
        for pitch in thisChord.pitchNames:
            chord_pitch.append(pitch)
        chord_label = chord.Chord(chord_pitch)
        allowed_chord_quality = ['incomplete major-seventh chord', 'major seventh chord',
                                 'incomplete minor-seventh chord', 'minor seventh chord',
                                 'incomplete half-diminished seventh chord', 'half-diminished seventh chord',
                                 'diminished seventh chord',
                                 'incomplete dominant-seventh chord', 'dominant seventh chord',
                                 'major triad',
                                 'minor triad',
                                 'diminished triad']
        if any(each in chord_label.pitchedCommonName for each in allowed_chord_quality):
            if harmony.chordSymbolFigureFromChord(chord_label).find('Identified') != -1:  # harmony.chordSymbolFigureFromChord cannot convert pitch classes into chord name sometimes, and the examples are below
                #print('debug')
                if chord_label.pitchedCommonName.find('-diminished triad') != -1: # chord_label.pitchedCommonName is another version of the chord name, but usually I cannot use it to get harmony.ChordSymbol to get pitch classes, so I translate these cases which could be processed by harmony.ChordSymbol later on
                    chord_name = chord_label.pitchedCommonName.replace('-diminished triad', 'o') # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete half-diminished seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-incomplete half-diminished seventh chord', '/o7') # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete minor-seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-incomplete minor-seventh chord', 'm7') # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete major-seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-incomplete major-seventh chord', 'M7') # translate to support
                elif chord_label.pitchedCommonName.find('-incomplete dominant-seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-incomplete dominant-seventh chord', '7') # translate to support
                elif chord_label.pitchedCommonName.find('-major triad') != -1: #(e.g., E--major triad) in  279 slice 33
                    chord_name = chord_label.pitchedCommonName.replace('-major triad', '') # translate to support
                elif chord_label.pitchedCommonName.find('-dominant seventh chord') != -1: #(e.g., E--major triad) in  279 slice 33
                    chord_name = chord_label.pitchedCommonName.replace('-dominant seventh chord', '7')# translate to support
                elif chord_label.pitchedCommonName.find('-half-diminished seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-half-diminished seventh chord', '/o7') # translate to support
                elif chord_label.pitchedCommonName.find('-minor-seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-minor-seventh chord', 'm7') # translate to support
                elif chord_label.pitchedCommonName.find('-major-seventh chord') != -1:
                    chord_name = chord_label.pitchedCommonName.replace('-major-seventh chord', 'M7') # translate to support
                else:
                    chord_name = chord_label.pitchedCommonName  # Just in case the function cannot accept any names (e.g., E--major triad)
            else:
                if chord_label.pitchedCommonName.find('-incomplete dominant-seventh chord') != -1: # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', chord_label.pitchedCommonName.replace('-incomplete dominant-seventh chord', '7'))  # remove 'add' part
                elif chord_label.pitchedCommonName.find('-incomplete major-seventh chord') != -1: # contains "add" which does not work for harmony.ChordSymbol. This is probably becasue G D F, lacking of third to be 7th chord, and it is wrongly identified as GpoweraddX, so it needs modification.
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', chord_label.pitchedCommonName.replace('-incomplete major-seventh chord', 'M7'))  # remove 'add' part
                elif harmony.chordSymbolFigureFromChord(chord_label).find('add') != -1: # contains "add" which does not work for harmony.ChordSymbol, at 095
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[:harmony.chordSymbolFigureFromChord(chord_label).find('add')]) # remove 'add' part
                # elif harmony.chordSymbolFigureFromChord(chord_label).find('power') != -1: # assume power alone as major triad
                #     chord_label_list.append(
                #         re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)[
                #                                        :harmony.chordSymbolFigureFromChord(chord_label).find(
                #                                            'power')]))  # remove 'add' part
                elif harmony.chordSymbolFigureFromChord(chord_label).find('dim') != -1:
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label).replace('dim','o'))
                else:
                    chord_name = re.sub(r'/[A-Ga-g][b#-]*', '', harmony.chordSymbolFigureFromChord(chord_label)) # remove inversions, notice that half diminished also has /!
                # the line above is the most cases, where harmony.chordSymbolFigureFromChord can give a chord name for the pitch classes, and Bdim is generated by this!
            thisChord.addLyric(chord_name)
        else:
            thisChord.addLyric(' ')


def extract_FB_as_lyrics():
    for filename in os.listdir(os.path.join('.', '371_FB')):
        if 'FB.musicxml' not in filename: continue
        #if '017' not in filename: continue
        print(filename)
        tree = ET.ElementTree(file=os.path.join('.', '371_FB', filename))
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
                        for each_FB_digit in ele.iter():
                            if each_FB_digit.tag == 'figure':
                                single_digit = adding_XXXfix(each_FB_digit, 'figure-number', single_digit)
                                single_suffix = adding_XXXfix(each_FB_digit, 'suffix', single_suffix)
                                single_prefix = adding_XXXfix(each_FB_digit, 'prefix', single_prefix)
                                FB_digit['number'] = single_digit
                                FB_digit['suffix'] = single_suffix
                                FB_digit['prefix'] = single_prefix
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
        tree.write(open(os.path.join('.', '371_FB', filename[:-9] + '_' + 'lyric' + '.xml'), 'w'), encoding='unicode')



def lyrics_to_chordify(want_IR):
    for filename in os.listdir(os.path.join('.', '371_FB')):
        if 'lyric' not in filename: continue
        elif 'chordify' in filename: continue
        print(filename)
        s = converter.parse(os.path.join('.', '371_FB', filename))
        bassline = s.parts[-1]
        sChords = s.chordify()
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
            thisChord.closedPosition(forceOctave=4, inPlace=True)
            each_measure = bassline.measure(thisChord.measureNumber)
            for each_bass in bassline.measure(thisChord.measureNumber).getElementsByClass(note.Note):
                if each_bass.beat == thisChord.beat:
                    bassnote = each_bass
                    if bassnote.lyrics != []:
                        fig = decode_FB_from_lyrics(bassnote.lyrics)
                        print(fig)
                        for j, one_FB in enumerate(fig):
                            translate_FB_into_chords(fig[j]['number'], sChords.recurse().getElementsByClass('Chord')[i + j], i + j)
                            for line in fig[j]['number']:
                                sChords.recurse().getElementsByClass('Chord')[i + j].addLyric(line)
                    break
            if bassnote.lyrics == []:  # slices without FB, it still needs a chord label
                translate_FB_into_chords('', thisChord, i)


        s.insert(0, sChords)
        if want_IR:
            IR = s.chordify()
            for c in IR.recurse().getElementsByClass('Chord'):
                c.closedPosition(forceOctave=4, inPlace=True)
                c.annotateIntervals()
            s.insert(0, IR)

        s.write('musicxml', os.path.join('.', '371_FB', filename[:-4] + '_' + 'chordify' + '.xml'))


if __name__ == '__main__':
    want_IR = True
    #extract_FB_as_lyrics()
        # till this point, all FB has been extracted and attached as lyrics underneath the bass line!
    lyrics_to_chordify(want_IR)

