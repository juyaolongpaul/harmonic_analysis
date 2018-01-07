from music21 import *
import os
format = ['mid']
'''for fn in os.listdir('.\\midi_files\\original\\'):
    
        if fn[-3:] == 'mid':

            s = converter.parse(os.getcwd()+'\\midi_files\\original\\'+fn)
            k = s.analyze('key')
            if k.mode == 'minor' :
                i = interval.Interval(k.tonic, pitch.Pitch('A'))
            else:
                i = interval.Interval(k.tonic, pitch.Pitch('C'))
            print(i)
            sNew = s.transpose(i)
            #sNew.write('midi', os.getcwd() + '\\midi_files\\' + fn)'''

for fn in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\'):

    if fn[-3:] == 'xml':

        s = converter.parse('.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\' + fn)
        k = s.analyze('key')
        if k.mode == 'minor':
            i = interval.Interval(k.tonic, pitch.Pitch('A'))
        else:
            i = interval.Interval(k.tonic, pitch.Pitch('C'))
        print(i)
        sNew = s.transpose(i)
        sNew.write('musicxml', '.\\bach-371-chorales-master-kern\\kern\\'+ 'transposed_' + fn[:-3] + 'xml')