from music21 import *
import os
import re
from transpose_to_C_chords import get_displacement
def transpose_polyphony(source, input):
    c1=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    #for fn in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\'):
    if source != 'Rameau':
        format = '.xml'
    else:
        format = '.mid'
    for fn in os.listdir(input):
        p = re.compile(r'\d{3}')  # find 3 digit in the file name
        id_id = p.findall(fn)
        if len(id_id) == 0:
            continue
        if (os.path.isfile(os.path.join(input, 'transposed_') + 'KBcKE' + id_id[0] + format) or os.path.isfile(os.path.join(input, 'transposed_') + 'KBc_oriKE' + id_id[0] + format)):
            continue
        print(os.path.join(input, 'transposed_') + 'KBcKE' + id_id[0] + format)
        if fn[-3:] == 'krn' or fn[-3:] == 'mid':  # we want to transpose krn file into musicxml file
            print(fn)
            s = converter.parse(os.path.join(input, fn))
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
                if i.directedName == 'P1':
                    key_name += '_ori'
                sNew = s.transpose(i)
                if(source == 'Rameau'):
                    sNew.write('midi', os.path.join(input,'transposed_') + 'KB' + key_name + 'KE' + id_id[0] + format)
                else:
                    sNew.write('musicxml', os.path.join(input,'transposed_') + 'KB' + key_name + 'KE' + id_id[0] + format)  # convert krn into xml

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