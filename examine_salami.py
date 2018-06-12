import os
import shutil
from music21 import *
cwd = '.\\bach_chorales_scores\\original_midi+PDF\\'


def examine_salami(source='DLfM2017', sign=0):
    """

    :param source:
    :param input:
    :return:
    """
    useless_files_number = 0
    useless_files_number_with_not = 0
    if(sign == 0):
        flog = open('salami_log_one+_difference' + source + '.txt', 'w')
    else:
        flog = open('salami_log_one+_difference' + source + '_transposed.txt', 'w')
    #sign = 0
    for fn in os.listdir(cwd):
        if fn[-3:] == 'mid':
            print(fn)
            if(source=='DLfM2017'):
                s = converter.parse(cwd + fn)  # we just used midi version
            elif(source=='Rameau'):  # We need to use krn version of the files
                if (os.path.isfile('.\\bach-371-chorales-master-kern\\kern\\' + 'transposed_KBcKEchor' + fn[0:3] + '.xml')):
                    s = converter.parse( '.\\bach-371-chorales-master-kern\\kern\\' + 'transposed_KBcKEchor' + fn[0:3] + '.xml')
                else:
                    continue
            sChords = s.chordify()
            input_slice_counter = 0
            for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                input_slice_counter += 1
            print(input_slice_counter)

            if (sign == 1):

                if (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[
                                                                                                                  0:3] + '.pop''')):
                    f = open(
                        '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop',
                        'r')
                    file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[
                                                                                                               0:3] + '.pop'
                elif (os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[
                                                                                                                    0:3] + '.pop.not''')):
                    f = open(
                        '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[
                                                                                                       0:3] + '.pop.not',
                        'r')
                    file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[
                                                                                                               0:3] + '.pop.not'
            else:
                if source == 'DLfM2017':
                    if (os.path.isfile(
                            '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                0:3] + '.pop''')):
                        f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                     0:3] + '.pop',
                                 'r')
                        file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                        0:3] + '.pop'
                    elif (os.path.isfile(
                            '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                0:3] + '.pop.not''')):
                        f = open(
                            '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                0:3] + '.pop.not',
                            'r')
                        file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_' + fn[
                                                                                                        0:3] + '.pop.not'
                elif source == 'Rameau':
                    if (os.path.isfile(
                            '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Rameau\\' + 'translated_' + fn[
                                                                                                0:3] + '_' + source + '.txt')):
                        f = open( '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Rameau\\' + 'translated_' + fn[
                                                                                                0:3] + '_' + source + '.txt',
                                 'r')
                        file_name =  '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Rameau\\' + 'translated_' + fn[
                                                                                                0:3] + '_' + source + '.txt'
            chord_slices = 0
            for line in f.readlines():
                chord_slices += len(line.split())
            print('salami slices of chorales: ' + str(input_slice_counter))
            print('salami slices of chords: ' + str(chord_slices))
            if (abs(chord_slices - input_slice_counter) >= 1 and chord_slices != 0):
                print(fn, file=flog)
                print('salami slices of chorales: ' + str(input_slice_counter), file=flog)
                print('salami slices of chords: ' + str(chord_slices), file=flog)
            if (input_slice_counter != chord_slices):
                if (chord_slices != 0):
                    if (file_name[-3:] == 'not'):
                        useless_files_number_with_not += 1
                    useless_files_number += 1
                    # input('??')
            else:
                if not os.path.exists('useful_chord_symbols'):
                    os.mkdir('useful_chord_symbols')
                    os.chdir('useful_chord_symbols')
                # shutil.copy(file_name, './')
    print('useless:' + str(useless_files_number), file=flog)
    print('useless with pop not:' + str(useless_files_number_with_not), file=flog)
if __name__ == "__main__":
        # Get input features

    examine_salami(sign=1)


