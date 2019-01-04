from music21 import *
import os
import matplotlib.pyplot as plt
import numpy as np
cwd = '.\\bach_chorales_scores\\transposed_MIDI\\'


def pitch_distribution(list, midi_num_freq):
    """
    Calculate the frequency of each note across all Bach chorales
    :param list:
    :param midi_num_freq:
    :return:
    """
    for i in list:
        midi_num_freq[i.midi - 1] += 1 # since the min number of midi is 1
        #if(i.midi > 84):
            #input('pitch class more than 84')
    return midi_num_freq

midi_num_freq = [0] * 128
midi_num_sequence = []
for id, fn in enumerate(os.listdir(cwd)):
    #if id>50: break
    print(fn)
    s = converter.parse(cwd + fn)
    sChords = s.chordify()
    for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
        for j in thisChord.pitches:
            midi_num_freq[j.midi - 1] += 1 # since the min number of midi is 1
            midi_num_sequence.append(j.midi)
for i, item in enumerate(midi_num_freq):
    print('midi number:', i+1, ' occurance:', item)
hist, bin_edges = np.histogram(midi_num_sequence, bins=len(midi_num_freq))
plt.bar(bin_edges[:-1], hist, width=1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.show()
