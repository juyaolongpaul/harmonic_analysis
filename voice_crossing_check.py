import os
from music21 import *
from get_input_and_output import get_pitch_class_for_four_voice

input = os.path.join('.', 'bach-371-chorales-master-kern', 'kern')
for file_name in os.listdir(input):
    if file_name[-4:] == '.krn':
        #if '086' not in file_name: continue
        print(file_name)
        s = converter.parse(os.path.join(input, file_name))
        sChords = s.chordify(removeRedundantPitches=False)
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
            # if i == 46:
            #     print('debug')
            pitch_class_four_voice, pitch_four_voice = get_pitch_class_for_four_voice(thisChord, s)
            for j, item in enumerate(pitch_four_voice):
                if pitch_class_four_voice[j] != -1 and pitch_class_four_voice[-1] != -1: # No rest in these two voices
                    if item.pitch.midi < pitch_four_voice[-1].pitch.midi: # any voice that has a lower pitch than bass
                        print('Measure number', thisChord.measureNumber, 'offset', thisChord.offset, 'bass note:', pitch_four_voice[-1].nameWithOctave, 'voice crossing is with', j)
