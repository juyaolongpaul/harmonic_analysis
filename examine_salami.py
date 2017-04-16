import os
import shutil
from music21 import *
cwd = '.\\bach_chorales_scores\\original_midi+PDF\\'

if __name__ == "__main__":
    if __name__ == "__main__":
        # Get input features
        useless_files_number = 0
        useless_files_number_with_not = 0
        for fn in os.listdir(cwd):
            if fn[-3:] == 'mid':
                print(fn)
                s = converter.parse(cwd + fn)
                sChords = s.chordify()
                print(len(sChords.notes))




                if(os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop''')):
                    f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop', 'r')
                    file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop'
                elif(os.path.isfile('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop.not''')):
                    f = open(
                        '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop.not',
                        'r')
                    file_name = '.\\genos-corpus\\answer-sheets\\bach-chorales\\' + 'translated_transposed_' + fn[0:3] + '.pop.not'

                chord_slices = 0
                for line in f.readlines():
                    chord_slices += len(line.split())
                print('salami slices of chorales: ' + str(len(sChords.notes)))
                print('salami slices of chords: ' + str(chord_slices))
                if(len(sChords.notes) != chord_slices):
                    if(chord_slices != 0):
                        if(file_name[-3:] == 'not'):
                            useless_files_number_with_not += 1
                        useless_files_number += 1
                        #input('??')
                else:
                    if not os.path.exists('useful_chord_symbols'):
                        os.mkdir('useful_chord_symbols')
                        os.chdir('useful_chord_symbols')
                    shutil.copy(file_name, './')
        print('useless:' + str(useless_files_number))
        print('useless with pop not:' + str(useless_files_number_with_not))

