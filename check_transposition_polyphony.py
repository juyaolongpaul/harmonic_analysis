import os
from music21 import *
for fn in os.listdir('.\\midi_files\\'):
    if fn[-3:] == 'mid':

        s = converter.parse(os.getcwd() + '\\midi_files\\' + fn)
        k = s.analyze('key')
        if(k.tonic._step == 'C' and k.mode == 'major') or (k.tonic._step == 'A' and k.mode == 'minor'):
            print(fn)
            print('correct')
        else:
            print(k.tonic._step+k.mode)
            print(fn)
            input('???')