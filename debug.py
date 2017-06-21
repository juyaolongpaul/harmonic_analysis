from music21 import *
import os
for file_name in os.listdir('.\\bach_chorales_scores\\transposed_MIDI\\'):
    print(file_name)
    s = converter.parse('.\\bach_chorales_scores\\transposed_MIDI\\' + file_name)
    k = s.analyze('key')
    print(k.name)
    if(k.name != 'C major' and k.name != 'A minor'):
        print('error:' + k.name)
