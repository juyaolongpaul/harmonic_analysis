# this file offers code realizing certain functions
import os
from music21 import *
def count_pickup_measure_NO():
    for fn in os.listdir(r'C:\Users\juyao\Documents\Github\harmonic_analysis\Bach_chorale_FB\FB_source\musicXML_master'):
        if 'FB_align' not in fn: continue
        print(fn)
        s = converter.parse(os.path.join(r'C:\Users\juyao\Documents\Github\harmonic_analysis\Bach_chorale_FB\FB_source\musicXML_master', fn))
        previous_beat = 0
        previous_mm_number = 0
        for i, thisChord in enumerate(s.parts[-1].recurse().getElementsByClass('Chord')):
            #print('measure number', thisChord.measureNumber, thisChord, 'beat:', thisChord.beat)
            if thisChord.measureNumber == previous_mm_number:
                if thisChord.beat < previous_beat:
                    print('pick up measure found!')
                previous_beat = thisChord.beat
            else:
                previous_beat = 0
                previous_mm_number = thisChord.measureNumber
    # this one counts the number of pick up measures and output them


def compare_chord_labels():
    input_path_array = []
    inputpath_alignment = os.path.join(os.getcwd(), 'new_music', 'New_alignment', 'predicted_result', 'original_key')
    inputpath_revised = os.path.join(os.getcwd(), 'new_music', 'New_revised', 'predicted_result', 'original_key')
    input_path_array.append(inputpath_alignment)
    input_path_array.append(inputpath_revised)
    for folder in os.listdir(input_path_array[0]):
        if os.path.isdir(os.path.join(input_path_array[0], folder)):
            for fn in os.listdir(os.path.join(input_path_array[0], folder)):
                if fn[-3:] == 'txt':
                    f1 = open(os.path.join(input_path_array[0], folder, fn))
                    f2 = open(os.path.join(input_path_array[1], folder, fn))
                    result1 = f1.readlines()
                    result2 = f2.readlines()
                    number_of_differences = 0
                    for id, each_result in enumerate(result1):
                        if each_result != result2[id]:
                            number_of_differences += 1
                    print('% of difference for', fn, 'is:', number_of_differences/len(result1))


if __name__ == "__main__":
    compare_chord_labels()
    #count_pickup_measure_NO()

