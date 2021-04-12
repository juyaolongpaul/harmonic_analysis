# this file offers code realizing certain functions
import os
from music21 import *
from scipy import stats
import numpy as np
from collections import Counter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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


def put_chords_into_files(sChord, a_chord_label=[], replace='Y', f=[]):
    previous_chord = []
    all_chords = []
    sign = 0 # judge whether the first chord is empty or not
    for i, thisChord in enumerate(sChord.recurse().getElementsByClass('Chord')):
        # obtain all the chord labels
        chord_label = []
        for j, label in enumerate(thisChord.lyrics):
            if len(label.text) > 0:
                if label.text[0].isalpha() and label.text[0].isupper(): # this is a chord label
                    chord_label.append(label.text)
                    sign = 1
                elif j == len(thisChord.lyrics) - 1 and sign == 0:
                    chord_label.append('n/a')
                    print('this is the special "chord"')
        if i == 0 and (len(chord_label) == 0):
            print('the first chord is empty!')
        elif len(chord_label) == 0:
            if not previous_chord == []:
                if f != []:
                    print(previous_chord, file=f)
                a_chord_label.append(previous_chord)
                all_chords.append(previous_chord)
            # else:
            #     print('debug')
        else:
            if f != []:
                print(chord_label, file=f)
            a_chord_label.append(chord_label)
            all_chords.append(chord_label)
        if chord_label != []:
            previous_chord = chord_label
    return a_chord_label, all_chords


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
    if '2_op13_1_' in fn and keyword != 'omr':
        replace = fn.replace(stage, keyword).replace('C', 'a_ori')
    else:
        replace = fn.replace(stage, keyword)
    with open(os.path.join(inputpath, replace)) as f:
        json_dict = json.loads(f.read())
        json_dict = {float(k):v for k, v in json_dict.items()}
        index = list(json_dict.keys())
        index = [float(value) for value in index]
    return index, json_dict


def compare_chord_labels(inputpath, keyword1, keyword2, keyword3, keyword4):
    a_diff1 = []
    a_diff2 = []
    a_diff3 = []
    for fn in os.listdir(inputpath):
        if not os.path.isdir(os.path.join(inputpath, fn)):
            if fn[-3:] == 'txt' and 'omr' in fn:
                # if 'op44iii_1' in fn or 'op44iii_2' in fn:
                #     continue
                # if 'op44iii_2' not in fn:
                #     continue
                print(fn)
                # f1 = open(os.path.join(input_path_array[0], fn))
                # try:
                #     f2 = open(os.path.join(input_path_array[1], fn))
                # except:
                #     f2 = open(os.path.join(input_path_array[1], fn.replace(keyword1, keyword2)))
                # json_dict1 = json.loads(f1)
                # json_dict2 = json.loads(f1)
                filename, stage = parse_filename(fn.strip())
                try:
                    get_index(fn, stage, keyword1)
                    index1, dict1 = get_index(fn, stage, keyword1)
                except:
                    continue
                try:
                    get_index(fn, stage, keyword2)
                    index2, dict2 = get_index(fn, stage, keyword2)
                except:
                    continue
                try:
                    get_index(fn, stage, keyword3)
                    index3, dict3 = get_index(fn, stage, keyword3)
                except:
                    continue
                try:
                    get_index(fn, stage, keyword4)
                    index4, dict4 = get_index(fn, stage, keyword4)
                except:
                    continue
                # index1, dict1= get_index(fn, stage, keyword1)
                # index2, dict2= get_index(fn, stage, keyword2)
                # index3, dict3= get_index(fn, stage, keyword3)
                # index4, dict4= get_index(fn, stage, keyword4)
                shared_index = index1
                shared_index = list(sorted(set(shared_index + index2)))
                shared_index = list(sorted(set(shared_index + index3)))
                shared_index = list(sorted(set(shared_index + index4)))
                whole_dict = {'shared_index':shared_index}
                whole_dict.update({keyword1:dict1})
                whole_dict.update({keyword2: dict2})
                whole_dict.update({keyword3: dict3})
                whole_dict.update({keyword4: dict4})
                # should make a dictionary here
                df = pd.DataFrame(whole_dict, index=whole_dict['shared_index'])
                # print(df)
                df.fillna(method='ffill', inplace=True)
                # print(df)
                orders = ['omr_corrected', 'corrected_revised', 'revised_aligned']
                df['omr_corrected'] = (df['omr'] != df['corrected'])
                df['corrected_revised'] = (df['corrected'] != df['revised'])
                df['revised_aligned'] = (df['revised'] != df['aligned'])
                diff = df['omr_corrected'].mean()
                diff2 = df['corrected_revised'].mean()
                diff3 = df['revised_aligned'].mean()
                df2 = df

                df = df.melt(id_vars=['shared_index'], value_vars=orders, var_name='comparison',
                             value_name='changed')
                df = df.astype({'changed': 'float64'})
                sns.relplot(
                    x='shared_index',
                    y='changed',
                    row='comparison',
                    kind='line',
                    height=1.5,
                    aspect=15.0,
                    data=df
                )
                plt.title(fn)
                # plt.show()
                print('comparison', diff, diff2, diff3)
                a_diff1.append(diff * 100)
                a_diff2.append(diff2 * 100)
                a_diff3.append(diff3 * 100)
    print('difference between OMR and CORRECTED:', np.median(a_diff1), '%', '±', np.std(a_diff1), '%', sorted(a_diff1)[0], sorted(a_diff1)[-1])
    print('difference between CORRECTED and REVISED:', np.median(a_diff2), '%', '±', np.std(a_diff2), '%', sorted(a_diff2)[0], sorted(a_diff2)[-1])
    print('difference between REVISED and ALIGNED:', np.median(a_diff3), '%', '±', np.std(a_diff3), '%', sorted(a_diff3)[0], sorted(a_diff3)[-1])
                # df.cc.astype('category').cat.codes
                ############## Output results as chord label integers
                # df = df.melt(id_vars=['shared_index'], var_name='stage', value_name='chord_labels')
                # #
                # print(df)
                # df['code'] = pd.factorize(df['chord_labels'])[0]
                # print(df)
                # plt.figure(figsize=(25, 6))
                # sns.lineplot(x='shared_index', y='code', hue='stage', data=df)
                # plt.show()
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

def finding_chord_root(chord_name):
    if '#' in chord_name or '-' in chord_name:
        return chord_name[:2]
    else:
        return chord_name[0]


def key_invariant_pairs(each_pair):
        chord_root_1 = finding_chord_root(each_pair[0])
        chord_quality_1 = each_pair[0][len(chord_root_1):]
        if chord_quality_1 == '':
            chord_quality_1 = 'M'
        chord_root_2 = finding_chord_root(each_pair[1])
        chord_quality_2 = each_pair[1][len(chord_root_2):]
        if chord_quality_2 == '':
            chord_quality_2 = 'M'
        a_interval_1 = interval.Interval(noteStart=pitch.Pitch(chord_root_1), noteEnd=pitch.Pitch(chord_root_2))
        number1 = abs(a_interval_1.semitones)
        number2 = abs(a_interval_1.complement.semitones)
        if number1 < number2:
            if '-' in a_interval_1.directedSimpleName:
                return ','.join([chord_quality_2, chord_quality_1, a_interval_1.directedSimpleName.replace('-', '')])
            else:
                return ','.join([chord_quality_1, chord_quality_2, a_interval_1.directedSimpleName])
        else:
            if '-' in a_interval_1.complement.directedSimpleName:
                return ','.join([chord_quality_2, chord_quality_1, a_interval_1.complement.directedSimpleName.replace('-', '')])
            else:
                return ','.join([chord_quality_1, chord_quality_2, a_interval_1.complement.directedSimpleName])


def print_this_plot_comparison():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 40})
    labels = ['M', 'm', '7', 'm7', 'o', 'M7', '/o7', 'o7', '+']
    one_one =  [0.44436519258202567, 0.22032017752417182, 0.11657948961800603, 0.08178792201616739, 0.05444602948169282, 0.03685211602472658, 0.03312727849104454, 0.009193216040576954, 0.0033285782215882074] # quality distribution for BCMCL 1.1
    one_one_chord_type = {'D': 0.07370423204945316, 'G': 0.06435251228403867, 'A': 0.05801236329053733, 'C': 0.050245680773498176, 'F': 0.04763036931367887, 'E': 0.045015057853859564, 'B-': 0.03938817562212712, 'Bm': 0.03471231573941987, 'Gm': 0.03360278966555714, 'Em': 0.0315422412426692, 'Am': 0.03146298938025044, 'Dm': 0.02456807734981772, 'Cm': 0.023696306863211284, 'F#m': 0.020446980504041846, 'D7': 0.02036772864162308, 'E-': 0.020129973054366777, 'A7': 0.017276906007291173, 'E7': 0.016008876208590903, 'B': 0.016008876208590903, 'F#': 0.0145823426850531, 'other': 0.3172452052623236}
    one_zero_chord_type = {'D': 0.08412509897070466, 'G': 0.07412905779889153, 'A': 0.06769596199524941, 'C': 0.05760095011876484, 'F': 0.05423594615993666, 'E': 0.05374109263657957, 'B-': 0.04493269992082344, 'Bm': 0.039885193982581155, 'Gm': 0.0367181314330958, 'Am': 0.0367181314330958, 'Em': 0.035134600158353124, 'Cm': 0.027216943784639746, 'Dm': 0.027019002375296912, 'F#m': 0.023258115597783055, 'E-': 0.022268408551068885, 'B': 0.019002375296912115, 'F#': 0.016330166270783847, 'D7': 0.01157957244655582, 'A7': 0.011282660332541567, 'G7': 0.010095011876484561, 'other': 0.24703087885985753}
    one_zero = [0.5113951644867222, 0.24900911613158938, 0.07520808561236624, 0.05648038049940547, 0.0587594133967499, 0.016845025762980578, 0.023285770907649623, 0.006936187078874356, 0.002080856123662307] # quality distribution for BCMCL 1.0
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, one_one, width, label='BCMCL 1.1')
    rects2 = ax.bar(x + width / 2, one_zero, width, label='BCMCL 1.0')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Chord Qualities')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    # plt.show()
    plt.savefig('chord_quality.png', dpi=300)



def print_this_plot_comparison_2():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 40})

    # one_one =  [0.44436519258202567, 0.22032017752417182, 0.11657948961800603, 0.08178792201616739, 0.05444602948169282, 0.03685211602472658, 0.03312727849104454, 0.009193216040576954, 0.0033285782215882074] # quality distribution for BCMCL 1.1
    one_one_chord_type = {'D': 0.07370423204945316, 'G': 0.06435251228403867, 'A': 0.05801236329053733,
                          'C': 0.050245680773498176, 'F': 0.04763036931367887, 'E': 0.045015057853859564,
                          'B-': 0.03938817562212712, 'Bm': 0.03471231573941987, 'Gm': 0.03360278966555714,
                          'Em': 0.0315422412426692, 'Am': 0.03146298938025044, 'Dm': 0.02456807734981772,
                          'Cm': 0.023696306863211284, 'F#m': 0.020446980504041846, 'D7': 0.02036772864162308,
                          'E-': 0.020129973054366777, 'A7': 0.017276906007291173,
                          'B': 0.016008876208590903, 'F#': 0.0145823426850531, 'other': 0.3172452052623236}
    # one_one_chord_type_ori = {'D': 0.07370423204945316, 'G': 0.06435251228403867, 'A': 0.05801236329053733, 'C': 0.050245680773498176, 'F': 0.04763036931367887, 'E': 0.045015057853859564, 'B-': 0.03938817562212712, 'Bm': 0.03471231573941987, 'Gm': 0.03360278966555714, 'Em': 0.0315422412426692, 'Am': 0.03146298938025044, 'Dm': 0.02456807734981772, 'Cm': 0.023696306863211284, 'F#m': 0.020446980504041846, 'D7': 0.02036772864162308, 'E-': 0.020129973054366777, 'A7': 0.017276906007291173, 'E7': 0.016008876208590903, 'B': 0.016008876208590903, 'F#': 0.0145823426850531, 'other': 0.3172452052623236}
    one_zero_chord_type = {'D': 0.08412509897070466, 'G': 0.07412905779889153, 'A': 0.06769596199524941, 'C': 0.05760095011876484, 'F': 0.05423594615993666, 'E': 0.05374109263657957, 'B-': 0.04493269992082344, 'Bm': 0.039885193982581155, 'Gm': 0.0367181314330958, 'Em': 0.035134600158353124, 'Am': 0.0367181314330958, 'Dm': 0.027019002375296912, 'Cm': 0.027216943784639746, 'F#m': 0.023258115597783055, 'D7': 0.01157957244655582, 'E-': 0.022268408551068885, 'A7': 0.011282660332541567, 'B': 0.019002375296912115, 'F#': 0.016330166270783847,  'other': 0.24703087885985753}
    labels = list(one_one_chord_type.keys())
    # one_zero_chord_type_ori = {'D': 0.08412509897070466, 'G': 0.07412905779889153, 'A': 0.06769596199524941, 'C': 0.05760095011876484,
    #  'F': 0.05423594615993666, 'E': 0.05374109263657957, 'B-': 0.04493269992082344, 'Bm': 0.039885193982581155,
    #  'Gm': 0.0367181314330958, 'Am': 0.0367181314330958, 'Em': 0.035134600158353124, 'Cm': 0.027216943784639746,
    #  'Dm': 0.027019002375296912, 'F#m': 0.023258115597783055, 'E-': 0.022268408551068885, 'B': 0.019002375296912115,
    #  'F#': 0.016330166270783847, 'D7': 0.01157957244655582, 'A7': 0.011282660332541567, 'G7': 0.010095011876484561,
    #  'other': 0.24703087885985753}
    # one_zero = [0.5113951644867222, 0.24900911613158938, 0.07520808561236624, 0.05648038049940547, 0.0587594133967499, 0.016845025762980578, 0.023285770907649623, 0.006936187078874356, 0.002080856123662307] # quality distribution for BCMCL 1.0
    x = np.arange(len(list(one_one_chord_type.keys())))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, list(one_one_chord_type.values()), width, label='BCMCL 1.1')
    rects2 = ax.bar(x + width / 2, list(one_zero_chord_type.values()), width, label='BCMCL 1.0')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Chord Types')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    # plt.show()
    plt.savefig('chord_type.png', dpi=300)

def print_this_plot():
    from matplotlib.ticker import PercentFormatter
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    plt.rcParams.update({'font.size': 40})
    figure(num=None, figsize=(4, 6), facecolor='w', edgecolor='k')
    counter_fre =  {'M': 0.44436519258202567, 'm': 0.22032017752417182, '7': 0.11657948961800603, 'm7': 0.08178792201616739, 'o': 0.05444602948169282, 'M7': 0.03685211602472658, '/o7': 0.03312727849104454, 'o7': 0.009193216040576954, '+': 0.0033285782215882074}# quality distribution for BCMCL 1.1
    counter_fre = {'M': 0.5113951644867222, 'm': 0.24900911613158938, '7': 0.07520808561236624,
                   'm7': 0.05648038049940547, 'o': 0.0587594133967499, 'M7': 0.016845025762980578, '/o7': 0.023285770907649623,
                   'o7': 0.006936187078874356, '+': 0.002080856123662307}
    plt.bar(list(counter_fre.keys()), counter_fre.values(), width=1, color='g')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('Percentage (%)')
    plt.xlabel('Chord Qualities')
    # plt.xlabel('Multiple Interpretations')
    plt.xticks(rotation='vertical')
    # plt.figure(figsize=(20, 5))
    plt.show()


def print_BCMCL11():
    all_p = [0.4350934155149625, 0.5491773915990509, 0.6192173385360287, 0.6643541013923685, 0.6996819018854105, 0.7279062996196246, 0.7482884180352124, 0.7658500083068531, 0.7805541215708448, 0.795411582224504, 0.8068947317407836, 0.8167974778805982, 0.8252633903610965, 0.8338570571724816, 0.8409072132277705, 0.8490197248481868, 0.8542482350894257, 0.8593169564881386, 0.8649735495358069, 0.8637807241268571, 0.868846724854954, 0.8736016828694826, 0.8773714971344271, 0.8813279210732841, 0.7688001570654448, 0.7736641213820832, 0.7791156316106437, 0.78491323669299, 0.7886877375975896, 0.7936404132877718, 0.7977369726973842, 0.8023207244546752, 0.8059197140178916, 0.809435697914284, 0.8130607226772085, 0.8165752930930461, 0.8205823985756131, 0.8236596735941765, 0.8264171822646469, 0.8155384517135527, 0.8182211553501526, 0.8203394493667207, 0.8226047145230723, 0.825355867610981, 0.8282440069338612, 0.8315172063074083, 0.834575730484296, 0.8370455927569076, 0.8396122330977069, 0.8398758635866823]
    all_r = [0.9395493851011212, 0.9216605668691958, 0.9087038560700984, 0.8980522176571872, 0.890220556970133, 0.882539127375803, 0.8729896105753424, 0.8663747899113632, 0.8596253678720538, 0.8528339138436607, 0.8472422170138152, 0.8425916361784056, 0.8368813754613884, 0.8328134666008369, 0.8279429524145072, 0.822991404031091, 0.8169134278409824, 0.8108822495072623, 0.8060986581792677, 0.8099662982327972, 0.8046563972762181, 0.7984862271356006, 0.793653692399827, 0.7879867046141059, 0.8384358866489183, 0.8357355085769713, 0.8335348708433529, 0.8307424104929536, 0.8310537367414377, 0.8283595299921259, 0.8245864721511269, 0.8216570282716988, 0.8232527228609117, 0.8196576221807051, 0.8160150777475238, 0.8121582787113493, 0.8090971411249331, 0.8066169513564837, 0.803283220730329, 0.802360366064794, 0.8000513707664203, 0.7973333350581356, 0.7988844021536678, 0.7946177977016633, 0.7915067713219182, 0.7879522242328934, 0.7853833657793458, 0.7822538426747129, 0.7788937749084079, 0.7789872723434171]
    all_acc = [0.3590319534295959, 0.4758300943976912, 0.5432451880170348, 0.5827926650440297, 0.6132180425937576, 0.6340792029206733, 0.6462557578681475, 0.6606938655736082, 0.668340469090948, 0.6764624131485395, 0.6815053954785074, 0.6869287173273066, 0.6909404774281009, 0.6960920221105903, 0.6991528270167345, 0.7016495384818965, 0.7015552132928098, 0.702049159611809, 0.7035135655882068, 0.7048068774835952, 0.7046609452766625, 0.7036553538620638, 0.7030673890933415, 0.7019613440105521, 0.7015897843855299, 0.6999079208963485, 0.6991273381637338, 0.6962696918981961, 0.695093348566908, 0.6919236321551446, 0.6897750569981982, 0.6880421533600625, 0.6855850598756189, 0.682868510649206, 0.681543950312954, 0.6795366811350747, 0.6782501530057278, 0.6764404546309473, 0.6751198708815563, 0.673227395221082, 0.6722430507946612, 0.6702619939688627, 0.6690363346294788, 0.6677165021823925, 0.6652922585162575, 0.6631764650952056, 0.6624800910006037, 0.6610964318505187, 0.6603786551842, 0.659053696492468]
    all_inc_acc = [0.3748285385175311, 0.5038243588669119, 0.5802635402836182, 0.6288083898402286, 0.6667667370919663, 0.6952231312322494, 0.7152371044842, 0.7360371427466055, 0.749790833624183, 0.7653992660972236, 0.7768586301545294, 0.7875716661477931, 0.796254487397824, 0.8070260672751832, 0.8156991717821669, 0.8244528102713697, 0.8312406452987006, 0.8376886564435722, 0.8455129954331502, 0.8530237382546648, 0.8586580289559331, 0.864782831220414, 0.870121181545376, 0.8743938396084878, 0.8797248633679764, 0.8834740558526416, 0.8879014763784563, 0.8911705590024903, 0.894486305257543, 0.8969583836501915, 0.8991781549084472, 0.9007524267487887, 0.9013914593814969, 0.9021855358891366, 0.9027845282866009, 0.9020884062350548, 0.9029719606788156, 0.9030421070751359, 0.902993499847961, 0.9020874549251866, 0.9008735009368714, 0.8988502161055154, 0.8972382376341319, 0.8946331356010797, 0.8909773033438283, 0.8883250475454274, 0.8859588808394511, 0.8817141618661279, 0.8769476136706235, 0.8710782334425676]
    all_threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    fig, axes = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
    axes.plot(all_threshold, all_acc, label='Subset Accuracy')  # 传递label参数
    axes.plot(all_threshold, all_inc_acc, label='Inclusive Accuracy')
    axes.legend(loc='best')
    plt.ylabel('Performance')
    plt.xlabel('Threshold Value')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()

def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


def debug():
    unit = ['D7, F#o', 'B,B7', 'B7,B', 'C,Cm7', 'D,DM7', "DM7,D", 'A,A7', 'B, B7', 'A, A7','A7, A', 'F#o, D7']
    for i, each_item in enumerate(unit):
        elements = each_item.split(',')
        for ii, each_chord in enumerate(elements):
            elements[ii] = elements[ii].replace(' ', '')
        unit[i] = ','.join(sorted(elements))
    print(unit)

if __name__ == "__main__":
    # inputpath = os.path.join(os.getcwd(), 'new_music', 'New_later', 'predicted_result')
    # compare_chord_labels(inputpath, 'omr', 'corrected', 'revised', 'aligned')
    # #count_pickup_measure_NO()
    #print_this_plot()
    # print_BCMCL11()
    print_this_plot_comparison()
    # key_invariant_pairs()