from music21 import *
import os
from transpose_to_C_chords import get_displacement
def transpose_polyphony():
    format = ['mid']
    c1=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    #for fn in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\'):
    for fn in os.listdir('.\\bach-371-chorales-master-kern\\kern\\'):
        if (os.path.isfile('.\\bach-371-chorales-master-kern\\kern\\'+ 'transposed_' + 'KBcKE' + fn[:-3] + 'xml')):
            break
        if fn[-3:] == 'krn':
            print(fn)
            s = converter.parse('.\\bach-371-chorales-master-kern\\kern\\' + fn)
            k = s.analyze('key')
            displacement = get_displacement(k)

            #print(i)

            for key_transpose in range(12):
                if k.mode == 'minor':
                    i = interval.Interval(k.tonic, pitch.Pitch(c1[(displacement - key_transpose - 3) % len(c1)]))
                else:
                    i = interval.Interval(k.tonic, pitch.Pitch(c1[displacement - key_transpose]))
                print(i)
                key_name = c1[(displacement - key_transpose) % len(c1)]

                sNew = s.transpose(i)
                sNew.write('musicxml', '.\\bach-371-chorales-master-kern\\kern\\'+ 'transposed_' + 'KB' + key_name + 'KE' + fn[:-3] + 'xml')  # convert krn into xml

if __name__ == "__main__":
    transpose_polyphony()
'''
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
'''