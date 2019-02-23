from music21 import *
import os
import re
c1=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
c2=['c','d-','d','e-','e','f','g-','g','a-','a','b-','b']


def get_displacement(k):
    """
    Get displacement, which specifies how many steps to move to C major or A minor
    :param k: key class from music 21
    :return: displcement
    """
    ptr = k.name.find(' ')
    key_tonic = k.name[:ptr]
    key_tonic = key_tonic.lower()
    #print('key=' + key_tonic)
    if key_tonic in c2:
        #print(c2.index(key_tonic))
        if (k.mode == 'major'):
            displacement = c2.index(key_tonic)
        else:
            displacement = (c2.index(key_tonic) + 3) % len(c2)
    elif key_tonic in c1:
        if (k.mode == 'major'):
            displacement = c1.index(key_tonic)
        else:
            displacement = (c1.index(key_tonic) + 3) % len(c1)
    else:
        print('pitch class can not be found!')
        input('what do you think about it')
    return displacement


def transpose(pitch_ori, transposed_interval):
    """
    Transpose the target pitch into the one with the key of C
    :param pitch: the pitch that needs to be transposed
    :return:
    """
    original_pitch = pitch.Pitch(pitch_ori)
    destination_pitch = original_pitch.transpose(transposed_interval)
    return destination_pitch.name#.replace('-', 'b')


def provide_path_12keys(input, f1, output, f2, source):
    """
    Provide the path for the input and output, and transpose the chorales into 12 keys
    :param input:
    :param output:
    :return:
    """
    print('Step 2: Transpose the chord annotations into 12 possible keys')
    import  re
    for file_name in os.listdir(output):
        if os.path.isfile(os.path.join(output, 'transposed_') + 'KBCKE' + file_name) or os.path.isfile(os.path.join(output, 'transposed_') + 'KBC_oriKE' + file_name) \
                or os.path.isfile(os.path.join(output, 'transposed_') + 'KBa_oriKE' + file_name) or os.path.isfile(os.path.join(output, 'transposed_') + 'KBaKE' + file_name):
            continue
        if file_name[-3:] == 'txt' and file_name.find('KB') == -1 and file_name.find('transposed') == -1 and file_name.find('translated') != -1:
                #if(file_name[:3] != '369'):
                    #continue
                if source == 'melodic':
                    ptr = file_name.find('translated_') + 10
                    s = converter.parse(os.path.join(input, file_name[ptr + 1:ptr + 4]) + f1)
                elif source == 'Rameau':
                    ptr = file_name.find('translated_') + 10
                    s = converter.parse(os.path.join('.', 'bach_chorales_scores', 'original_midi+PDF', file_name[ptr + 1:ptr + 4]) + '.mid') # Use ly version
                else:
                    p = re.compile(r'\d{3}')
                    ptr = p.findall(file_name)
                    s = converter.parse(os.path.join(input, 'chor') + ptr[0] + f1)
                k = s.analyze('key')

                #print('acc ' + str(k.tonic._accidental.alter))
                displacement = get_displacement(k)
                for key_transpose in range(12):
                    if k.mode == 'minor':
                        transposed_interval = interval.Interval(k.tonic, pitch.Pitch(c2[(displacement - key_transpose - 3) % len(c2)]))
                        key_name = c2[(displacement - key_transpose - 3) % len(c2)].lower()
                    else:
                        transposed_interval = interval.Interval(k.tonic, pitch.Pitch(c2[displacement - key_transpose]))
                        key_name = c2[(displacement - key_transpose) % len(c2)].upper()
                    if transposed_interval.directedName == 'P1':
                        key_name = key_name + '_ori'
                    f = open(os.path.join(output, file_name), 'r')
                    fnew = open(os.path.join(output, 'transposed_') + 'KB' + key_name + 'KE' + file_name, 'w')
                    #fexception = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\'+ 'log.txt', 'a+')
                    p = re.compile(r'[#-]+')
                    for line in f.readlines():
                        id_id = p.findall(line)
                        if id_id != []: # has flat or sharp
                            root_ptr = re.search(r'[#-]+', line).end() # get the last flat or sharp position
                            transposed_result = transpose(line[0: root_ptr], transposed_interval) + line[root_ptr:]
                            print(transposed_result, end='', file=fnew)
                            #print('original: ', line, 'transposed: ', transposed_result)
                        else: # no flat or sharp, which means only the first element is the root
                            transposed_result = transpose(line[0], transposed_interval) + line[1:]
                            print(transposed_result, end='', file=fnew)
                            #print('original: ', line, 'transposed: ', transposed_result)






