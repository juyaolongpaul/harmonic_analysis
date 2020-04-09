# this file offers code realizing certain functions
import os
from music21 import *
from sequence_alignment import AffineNeedlemanWunsch
from scipy import stats
import numpy as np
from collections import Counter
import json
import pandas as pd


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


def put_chords_into_files(sChord, f):
    for i, thisChord in enumerate(sChord.recurse().getElementsByClass('Chord')):
        chord_label = thisChord.lyrics[-1].text
        chord_label = chord_label.replace('?', '').replace('!', '')
        if i == 0 and (chord_label == ' ' or chord_label == ''):
            input('the first chord is empty!')
        elif chord_label == ' ' or chord_label == '':
            if not previous_chord == ' ' or previous_chord == '':
                print(previous_chord, file=f)
            # else:
            #     print('debug')
        else:
            print(chord_label, file=f)
        # if i>0:
        #     if len(previous_chord) == 0:
        #         print('debug')
        if chord_label != ' ' and chord_label != '':
            previous_chord = chord_label


def extract_chord_labels():
    path = os.path.join(os.getcwd(), 'Bach_chorale_FB', 'FB_source', 'musicXML_master')
    for fn in os.listdir(path):
        if 'chordify' not in fn:
            continue
        if 'FB.txt' in fn or 'IR.txt' in fn:
            continue
        print('extracting chord labels for', fn)
        f_FB = open(os.path.join(path, fn[:-4] + '_FB.txt'), 'w')
        f_IR = open(os.path.join(path, fn[:-4] + '_IR.txt'), 'w')
        s = converter.parse(os.path.join(path, fn))
        voice_FB = s.parts[-2]
        voice_IR = s.parts[-1]
        put_chords_into_files(voice_IR, f_IR)
        put_chords_into_files(voice_FB, f_FB)
        f_FB.close()
        f_IR.close()


def compare_against_sam():
    path_FB = os.path.join(os.getcwd(), 'genos-corpus', 'answer-sheets', 'bach-chorales', 'New_annotation', 'ISMIR2019', 'comparing_to_FB_translation', 'FB')
    path_IR = os.path.join(os.getcwd(), 'genos-corpus', 'answer-sheets', 'bach-chorales', 'New_annotation', 'ISMIR2019', 'comparing_to_FB_translation', 'IR')
    path_GT = os.path.join(os.getcwd(), 'genos-corpus', 'answer-sheets', 'bach-chorales', 'New_annotation', 'ISMIR2019', 'comparing_to_FB_translation', 'Sam')
    a_counter = 0
    a_counter_correct_FB = 0
    a_counter_correct_IR = 0
    FB_accuracy = []
    IR_accuracy = []
    FB_error = []
    IR_error = []
    for fn in os.listdir(path_GT):
        f_FB = open(os.path.join(path_FB, fn[:-4] + '_FB_lyric_chordify_FB.txt'))
        f_IR = open(os.path.join(path_IR, fn[:-4] + '_FB_lyric_chordify_IR.txt'))
        f_GT = open(os.path.join(path_GT, fn))
        FB_results = f_FB.readlines()
        IR_results = f_IR.readlines()
        GT_results = f_GT.readlines()
        for i, elem in enumerate(FB_results):
            if 'maj7' in elem:
                FB_results[i] = FB_results[i].replace('maj', 'M')
        for i, elem in enumerate(IR_results):
            if 'maj7' in elem:
                IR_results[i] = IR_results[i].replace('maj', 'M')
        if not len(FB_results) == len(IR_results) and len(FB_results) == len(GT_results):
            print('does not align!')
        counter = len(IR_results)
        counter_correct_FB = 0
        counter_correct_IR = 0
        for i, each_GT_label in enumerate(GT_results):
            if each_GT_label == FB_results[i]:
                counter_correct_FB += 1
                a_counter_correct_FB += 1
            else:
                FB_error.append((each_GT_label+':'+FB_results[i]).replace('\n', ''))
            if each_GT_label == IR_results[i]:
                counter_correct_IR += 1
                a_counter_correct_IR += 1
            else:
                IR_error.append((each_GT_label+':'+IR_results[i]).replace('\n', ''))
        print('FB accuracy for', fn, 'is', counter_correct_FB / counter)
        FB_accuracy.append((counter_correct_FB / counter) * 100)
        print('IR accuracy for', fn, 'is', counter_correct_IR / counter)
        IR_accuracy.append((counter_correct_IR / counter) * 100)
        a_counter += counter
    print('FB errors', Counter(FB_error), 'total count is', len(FB_error))
    c = Counter(FB_error)
    s = sum(c.values())
    for elem, count in c.items():
        print(elem, count / s)
    print('IR errors', Counter(IR_error), 'total count is', len(IR_error))
    c = Counter(IR_error)
    s = sum(c.values())
    for elem, count in c.items():
        print(elem, count / s)
    print('Overall FB accuracy for is', np.mean(FB_accuracy), '%', '±', stats.sem(FB_accuracy), '%')
    print('Overall FB accuracy for is', np.mean(IR_accuracy), '%', '±', stats.sem(IR_accuracy), '%')
    print('Overall FB accuracy for is', np.mean(FB_accuracy), '%', '±', np.std(FB_accuracy), '%')
    print('Overall FB accuracy for is', np.mean(IR_accuracy), '%', '±', np.std(IR_accuracy), '%')


def parse_filename(f):
    f = f.replace('.musi', '').rsplit('_', 3)
    filename, stage, a, b = f
    return filename, stage


def get_index(fn, stage, keyword):
    with open(os.path.join(inputpath, fn.replace(stage, keyword))) as f:
        json_dict1 = json.loads(f.read())
        index = list(json_dict1.keys())
        index = [float(value) for value in index]
    return index


def compare_chord_labels(inputpath, keyword1, keyword2, keyword3, keyword4):
    for fn in os.listdir(inputpath):
        if not os.path.isdir(os.path.join(inputpath, fn)):
            if fn[-3:] == 'txt':
                # f1 = open(os.path.join(input_path_array[0], fn))
                # try:
                #     f2 = open(os.path.join(input_path_array[1], fn))
                # except:
                #     f2 = open(os.path.join(input_path_array[1], fn.replace(keyword1, keyword2)))
                # json_dict1 = json.loads(f1)
                # json_dict2 = json.loads(f1)
                filename, stage = parse_filename(fn.strip())
                index1 = get_index(fn, stage, keyword1)
                index2 = get_index(fn, stage, keyword2)
                index3 = get_index(fn, stage, keyword3)
                index4 = get_index(fn, stage, keyword4)
                shared_index = index1
                shared_index = list(sorted(set(shared_index + index2)))
                shared_index = list(sorted(set(shared_index + index3)))
                shared_index = list(sorted(set(shared_index + index4)))
                # should make a dictionary here
                df = pd.DataFrame(shared_index)
                df.fillna(method='ffill', inplace=True)
                # result1 = f1.read().splitlines()
                # result2 = f2.read().splitlines()
                # number_of_differences = 0
                # if len(result1) != len(result2):
                #     print('-------------------------------------------')
                #     print('dimensions for', fn, 'is different!', 'f1 is', len(result1), 'f2 is', len(result2))
                #     print(AffineNeedlemanWunsch(result1, result2))
                #     break
                #     # s1 = converter.parse(os.path.join(os.getcwd(), 'new_music', 'New_alignment', fn.replace('musi_chord_labels.txt', 'musicxml')))
                #     # s2 = converter.parse(os.path.join(os.getcwd(), 'new_music', 'New_corrected', fn.replace('musi_chord_labels.txt', 'musicxml').replace('revised', 'corrected')))
                #     # s1_chordify = s1.chordify()
                #     # s2_chordify = s2.chordify()
                #     # print('music dimensions for f1 is', len(s1_chordify.recurse().getElementsByClass('Chord')), 'f2 is', len(s2_chordify.recurse().getElementsByClass('Chord')))
                # else:
                #     for id, each_result in enumerate(result1):
                #         if id < len(result2):
                #             if each_result != result2[id]:
                #                 number_of_differences += 1
                #
                # print('% of difference for', fn, 'is:', number_of_differences/len(result1))
                # if len(result1) != len(result2):
                #     print('-------------------------------------------')

if __name__ == "__main__":
    inputpath = os.path.join(os.getcwd(), 'new_music', 'New_later', 'predicted_result', 'original_key')
    compare_chord_labels(inputpath, 'omr', 'corrected', 'revised', 'aligned')
    #count_pickup_measure_NO()

