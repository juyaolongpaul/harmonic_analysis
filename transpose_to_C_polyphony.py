from music21 import *
import os
import re
from transpose_to_C_chords import get_displacement


def transpose_into_12_keys(fn, filename, source, input, format):
    """
    Modular function that translates one file into 12 keys
    :return:
    """
    c2 = ['c', 'd-', 'd', 'e-', 'e', 'f', 'f#', 'g', 'a-', 'a', 'b-', 'b']
    print(fn)
    s = converter.parse(os.path.join(input, fn))
    k = s.analyze('AardenEssen')
    displacement = get_displacement(k)

    # print(i)
    # transpose on the flat side
    for key_transpose in range(12):
        if k.mode == 'minor':
            i = interval.Interval(k.tonic, pitch.Pitch(c2[(displacement - key_transpose - 3) % len(c2)]))
            key_name = c2[(displacement - key_transpose - 3) % len(c2)].lower()
        else:
            i = interval.Interval(k.tonic, pitch.Pitch(c2[displacement - key_transpose]))
            key_name = c2[(displacement - key_transpose) % len(c2)].upper()
        print(i)
        # if key_name != 'C' and key_name != 'a':
        #     continue  # TODO: only the transposition only works for C major or A minor, for other keys, I need to do F## processing!
        if i.directedName == 'P1' or i.directedName == 'd2':  # account for pitch spelling
            key_name += '_ori'
        if os.path.exists(os.path.join(input, 'transposed_') + 'KB' + key_name + 'KE' + filename + format):
            continue  # if already exists, no need to transpose
        sNew = s.transpose(i)
        if (source == 'Rameau'):
            sNew.write('midi', os.path.join(input, 'transposed_') + 'KB' + key_name + 'KE' + filename + format)
        else:
            sNew.write('musicxml', os.path.join(input, 'transposed_') + 'KB' + key_name + 'KE' + filename + format)  # convert krn into xml


def transpose_polyphony(source, input, bach='Y'):
    """
    This function is translating for chord labeling
    :param source:
    :param input:
    :param bach:
    :return:
    """
    print('Step 3: Translate the music into 12 keys')

    #for fn in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\'):
    if source != 'Rameau':
        format = '.xml'
    else:
        format = '.mid'
    for fn in os.listdir(input):
        if bach == 'Y':
            p = re.compile(r'\d{3}')  # find 3 digit in the file name
            id_id = p.findall(fn)
            if len(id_id) == 0:
                continue
            if (os.path.isfile(os.path.join(input, 'transposed_') + 'KBCKE' + id_id[0] + format) or os.path.isfile(
                    os.path.join(input, 'transposed_') + 'KBC_oriKE' + id_id[0] + format)
                    or os.path.isfile(
                        os.path.join(input, 'transposed_') + 'KBa_oriKE' + id_id[0] + format) or os.path.isfile(
                        os.path.join(input, 'transposed_') + 'KBaKE' + id_id[0] + format)):
                continue  # TODO: Need to change the name of this later!
        else:
            id_id = []
            id_id.append(os.path.splitext(fn)[0])
            if 'transposed_' in fn:
                continue
        print(os.path.join(input, 'transposed_') + id_id[0] + format)
        if fn[-3:] == 'krn' or fn[-3:] == 'mid' or fn[-3:] == 'xml':  # we want to transpose krn file into musicxml file
            transpose_into_12_keys(fn, id_id[0], source, input, format)


def transpose_polyphony_FB(source, input):
    """
    This function is translating for chord labeling
    :param source:
    :param input:
    :param bach:
    :return:
    """
    print('Translate the music into 12 keys')
    format = '.xml'
    if not os.path.isdir(os.path.join(input, 'transpose')):
        os.mkdir(os.path.join(input, 'transpose'))
    for fn in os.listdir(input):
        id_id = []
        id_id.append(os.path.splitext(fn)[0])
        if 'transposed_' in fn:
            # do not have to transpose when it is already done
            continue
        if 'FB_align' not in fn:
            continue
        print(os.path.join(input, 'transposed_') + id_id[0] + format)
        if fn[-3:] == 'krn' or fn[-3:] == 'mid' or fn[-3:] == 'xml':  # we want to transpose krn file into musicxml file
            transpose_into_12_keys(fn, id_id[0], source, input, format)
    # for fn in os.listdir(input):
    #     if 'transposed' not in fn:
    #         continue
    #     os.rename(os.path.join(input, fn), os.path.join(input, 'transpose', fn))

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