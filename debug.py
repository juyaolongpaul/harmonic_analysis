from music21 import *
from get_input_and_output import get_pitch_class_for_four_voice
s = converter.parse(r'G:\Projects\harmonic_analysis\transposed_KBfKE328_test.xml')
sChords = s.chordify()
for thisChord in sChords.recurse().getElementsByClass('Chord'):
    if len(thisChord.pitches) < 4:
        list = get_pitch_class_for_four_voice(thisChord, s)
        print(list)
'''for thisChord in sChords.recurse().getElementsByClass('Chord'):
    if len(thisChord.pitches) < 4:
        for j, part in enumerate(s.parts):
            for i, note in enumerate(part.measure(thisChord.measureNumber).notes):

                if note.beatStr == thisChord.beatStr:
                    print(j, note.pitch)
                else:
                    if part.measure(thisChord.measureNumber).notes[i].beatStr < thisChord.beatStr and \
                            part.measure(thisChord.measureNumber).notes[i + 1].beatStr > thisChord.beatStr:
                        print(j, part.measure(thisChord.measureNumber).notes[i].pitch)   '''
