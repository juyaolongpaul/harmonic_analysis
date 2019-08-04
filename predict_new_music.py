from keras.models import load_model
from transpose_to_C_polyphony import transpose_polyphony
import os
from music21 import *
from get_input_and_output import *
from predict_result_for_140 import get_predict_file_name
from transpose_to_C_chords import transpose
from predict_result_for_140 import output_NCT_to_XML, infer_chord_label1, infer_chord_label2, infer_chord_label3, \
    unify_GTChord_and_inferred_chord
from test_musicxml_gt import translate_chord_name_into_music21


def generate_ML_matrix(path, windowsize, sign='N'):
    counter = 0
    fn_all = []  # Unify the order
    for fn in os.listdir(path):
        if sign == 'N':  # eliminate pitch class only encoding
            if fn.find('_pitch_class') != -1 or fn.find('_chord_tone') != -1:
                continue
        elif sign == 'Y':  # only want pitch class only encoding
            if fn.find('_pitch_class') == -1:
                continue
        elif sign == 'C':  # only want chord tone as input to train the chord inferral algorithm
            if fn.find('_chord_tone') == -1:
                continue
        if fn.find('CKE') == -1 and fn.find('C_oriKE') == -1 and fn.find('aKE') == -1 and fn.find(
                'a_oriKE') == -1:  # we cannot find key of c, skip
            continue
        # elif portion == 'valid' or portion == 'test': # we want original key on valid and test set when augmenting
        #     if fn.find('_ori') == -1:
        #         continue
        fn_all.append(fn)
    fn_all.sort()
    print(fn_all)
    for fn in fn_all:
        encoding = np.loadtxt(os.path.join(path, fn))
        encoding_window = adding_window_one_hot(encoding, windowsize)
        if counter == 0:
            encoding_all = list(encoding_window)
            encoding_all = np.array(encoding_all)
        else:
            encoding_all = np.concatenate((encoding_all, encoding_window))
        counter += 1
    print('finished')
    return encoding_all


def get_input_encoding(inputpath, encoding_path):
    augmentation = 'N'
    fn_total = []
    if not os.path.isdir(encoding_path):
        os.mkdir(encoding_path)

    for id, fn in enumerate(os.listdir(
            inputpath)):  # this part should be executed no matter what since we want a updated version of chord list
        if fn.find('C_ori') != -1:
            fn_total.append(fn)
        if fn.find('CKE') != -1:
            fn_total.append(fn)

        if fn.find('aKE') != -1:
            fn_total.append(fn)
        if fn.find('a_oriKE') != -1:  # only wants key c
            fn_total.append(fn)

    for id, fn in enumerate(fn_total):
        chorale_x = []
        # pitch class is used. This one is used to indicate which one is NCT.
        chorale_x_only_pitch_class = []
        chorale_x_only_meter = []  # we want to save the meter info for the gt chord label as input feature to do chord inferral
        chorale_x_only_newOnset = []
        print(fn, id)
        s = converter.parse(os.path.join(inputpath, fn))
        sChords = s.chordify(removeRedundantPitches=False)
        part = s.parts[0]
        s_new = stream.Stream()
        if len(part.recurse().getElementsByClass(meter.TimeSignature)) == 0:  # No time signature
            print('no time signature')
            for each_part in s.parts:
                each_part.insert(0, meter.TimeSignature('4/4'))
                s_new.append(each_part)
            s_new.write('musicxml',
                        fp=os.path.join(inputpath, fn))
            s = converter.parse(os.path.join(inputpath, fn))
            sChords = s.chordify(removeRedundantPitches=False)
        for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
            # print('slice ID:', i, 'and pitch classes are:', thisChord.pitchClasses)
            pitchClass = [0] * 12
            if inputpath.find('Schutz') != -1:  # New onset does not work in Schutz
                only_pitch_class, pitchClass, newOnset = fill_in_pitch_class(pitchClass, thisChord.pitchClasses,
                                                                             thisChord,
                                                                             '', s, sChords, i)
                pitchClass = pitchClass + newOnset
            else:
                only_pitch_class, pitchClass, newOnset = fill_in_pitch_class(pitchClass, thisChord.pitchClasses,
                                                                             thisChord,
                                                                             'NewOnset', s, sChords,
                                                                             i)  # New onset does not work in Schutz

            meters, pitchClass = add_beat_into(pitchClass, thisChord.beatStr, '3meter', thisChord.beatStrength)

            if (i == 0):
                chorale_x = np.concatenate((chorale_x, pitchClass))
                chorale_x_only_pitch_class = np.concatenate((chorale_x_only_pitch_class, only_pitch_class))
                chorale_x_only_meter = np.concatenate((chorale_x_only_meter, meters))
                chorale_x_only_newOnset = np.concatenate((chorale_x_only_newOnset, newOnset))
            else:
                chorale_x = np.vstack((chorale_x, pitchClass))
                chorale_x_only_pitch_class = np.vstack((chorale_x_only_pitch_class, only_pitch_class))
                chorale_x_only_meter = np.vstack((chorale_x_only_meter, meters))
                chorale_x_only_newOnset = np.vstack((chorale_x_only_newOnset, newOnset))
        file_name_x = os.path.join(encoding_path,
                                   fn[:-4] + '.txt')
        file_name_xx = os.path.join(encoding_path,
                                    fn[:-4] + '_pitch_class.txt')
        np.savetxt(file_name_x, chorale_x, fmt='%.1e')
        np.savetxt(file_name_xx, chorale_x_only_pitch_class, fmt='%.1e')


def predict_new_music(modelpath_NCT, modelpath_CL, modelpath_DH, inputpath, bach='N'):
    transpose_polyphony(inputpath, inputpath, 'N')  # tranpose to 12 keys
    encoding_path = os.path.join(inputpath, 'encodings')
    if not os.path.isdir(os.path.join(inputpath, 'encodings')):
        os.mkdir(os.path.join(inputpath, 'encodings'))
    get_input_encoding(inputpath, encoding_path)  # generate input encodings
    xx = generate_ML_matrix(encoding_path, 1)
    xx_no_window = generate_ML_matrix(encoding_path, 0)
    xx_only_pitch = generate_ML_matrix(encoding_path, 0, 'Y')
    xx_chord_tone = list(xx_only_pitch)
    model = load_model(modelpath_NCT)  # we need to assemble x now
    model_chord_tone = load_model(modelpath_CL)
    model_direct_harmonic_analysis = load_model(modelpath_DH)
    predict_y = model.predict(xx)
    predict_y_direct_harmonic_analysis = model_direct_harmonic_analysis.predict_classes(xx)
    for i in predict_y:  # regulate the prediction
        for j, item in enumerate(i):
            if (item > 0.5):
                i[j] = 1
            else:
                i[j] = 0
    for i, item in enumerate(xx_chord_tone):

        NewOnset = list(xx_no_window[i][12:24])  # we need the onset sign of the vector
        for j, item2 in enumerate(item):
            if int(predict_y[i][j]) == 1:  # predict_y predicts NCT label for each slice
                if int(item2) == 1:  # if the there is a current pitch class and it is predicted as a NCT
                    xx_chord_tone[i][
                        j] = 0  # remove this pitch class since predict_xx should be all predicted CT

                    NewOnset[j] = 0
                # else:
                #     input('there is a NCT for a non-existing pitch class?!')

        xx_chord_tone[i] = np.concatenate(
            (xx_chord_tone[i], NewOnset))

        xx_chord_tone[i] = np.concatenate(
            (xx_chord_tone[i], xx_no_window[i][-3:]))  # add beat feature
        # TODO: 3 might not be modular enough

    predict_xx_chord_tone_window = adding_window_one_hot(np.asarray(xx_chord_tone), 2)

    predict_y_chord_tone = model_chord_tone.predict_classes(predict_xx_chord_tone_window,
                                                            verbose=0)  # TODO: we need to make this part modular so it can deal with all possible specs

    fileName, numSalamiSlices = get_predict_file_name(inputpath, [], 'N', bach='N')
    length = len(fileName)
    with open('chord_name_retrained.txt') as f:
        chord_name = f.read().splitlines()
    a_counter = 0
    if not os.path.isdir(os.path.join(inputpath, 'predicted_result')):
        os.mkdir(os.path.join(inputpath, 'predicted_result'))
    if not os.path.isdir(os.path.join(inputpath, 'predicted_result', 'original_key')):
        os.mkdir(os.path.join(inputpath, 'predicted_result', 'original_key'))
    for i in range(length):
        num_salami_slice = numSalamiSlices[i]
        chord_label_list = []  # For RB chord inferred labels
        chord_tone_list = []  # store all the chord tones predicted by the model
        all_answers_per_chorale = [{} for j in range(10000)]
        print(fileName[i])
        s = converter.parse(os.path.join(inputpath, fileName[i]))
        s_ori = converter.parse(os.path.join(inputpath, fileName[i]))
        sChords = s.chordify()
        s.insert(0, sChords)
        # transpose back to the original keys
        if bach == 'Y':
            p = re.compile(r'\d{3}')
            id = p.findall(fileName[i])
        else:
            id = []
            id.append(os.path.splitext(fileName[i])[0])
        for fn in os.listdir(inputpath):
            if id[0] in fn:  # get the key info
                ptr = fn.find('KB')
                if '-' not in fn and '#' not in fn:  # keys do not have accidentals
                    key_info = fn[ptr + 2]
                else:
                    key_info = fn[ptr + 2: ptr + 4]
                if key_info.isupper():  # major key
                    mode = 'major'
                else:
                    mode = 'minor'
                if mode == 'minor':
                    transposed_interval = interval.Interval(pitch.Pitch('A'), pitch.Pitch(key_info))
                else:
                    transposed_interval = interval.Interval(pitch.Pitch('C'), pitch.Pitch(key_info))
                if not os.path.isdir(os.path.join(inputpath, 'predicted_result', 'original_key', key_info)):
                    os.mkdir(os.path.join(inputpath, 'predicted_result', 'original_key', key_info))
                sNew = s_ori.transpose(transposed_interval)
                sChords_new = sNew.chordify()
                sNew.insert(0, sChords_new)
                acc = re.compile(r'[#-]+')
                f_transposed = open(os.path.join(inputpath, 'predicted_result', fileName[i][
                                                                                :-4]) + '_chord_labels.txt', 'w')
                f_ori = open(os.path.join(inputpath, 'predicted_result', 'original_key', key_info, fn[
                                                                                                   :-4]) + '_chord_labels.txt',
                             'w')
                for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

                    print('current slice is:', j, 'DHA-d is:', len(predict_y_direct_harmonic_analysis), 'CT-d is:',
                          len(predict_y_chord_tone), 'a_counter is:', a_counter)
                    thisChord.closedPosition(forceOctave=4, inPlace=True)
                    sChords_new.recurse().getElementsByClass('Chord')[j].closedPosition(forceOctave=4, inPlace=True)
                    x = xx_only_pitch[a_counter]
                    chord_tone = output_NCT_to_XML(x, predict_y[a_counter], thisChord, '_pitch_class')
                    chord_tone_list, chord_label_list = infer_chord_label1(thisChord, chord_tone, chord_tone_list,
                                                                           chord_label_list)
                    if j == 0:
                        thisChord.addLyric(
                            ('NCT + CL (ML):', chord_name[predict_y_chord_tone[a_counter]]))
                        print('NCT + CL (ML):', chord_name[predict_y_chord_tone[a_counter]], file=f_transposed)
                        thisChord.addLyric(
                            ('DH (ML):', chord_name[predict_y_direct_harmonic_analysis[a_counter]]))
                    else:
                        thisChord.addLyric(
                            (chord_name[predict_y_chord_tone[a_counter]]))  # Output chords in the transposed key
                        thisChord.addLyric(chord_name[predict_y_direct_harmonic_analysis[a_counter]])
                        print(chord_name[predict_y_chord_tone[a_counter]], file=f_transposed)
                    all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                        chord_name[predict_y_direct_harmonic_analysis[a_counter]]))] = all_answers_per_chorale[j].get(
                        unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                            chord_name[predict_y_direct_harmonic_analysis[a_counter]])), 0) + 1
                    all_answers_per_chorale[j][unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                        chord_name[predict_y_chord_tone[a_counter]]))] = all_answers_per_chorale[j].get(
                        unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(
                            chord_name[predict_y_chord_tone[a_counter]])), 0) + 1

                    a_counter += 1
                for j, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):
                    if (chord_label_list[j] == 'un-determined' or chord_label_list[j].find(
                            'interval') != -1):  # sometimes the last
                        # chord is un-determined because there are only two tones!
                        infer_chord_label2(j, thisChord, chord_label_list, chord_tone_list)  # determine the final chord
                    infer_chord_label3(j, thisChord, chord_label_list,
                                       chord_tone_list)  # TODO: Look into this later: ch
                    if j == 0:
                        thisChord.addLyric(
                            ('NCT + CL (RB):', chord_label_list[j]))
                    else:
                        thisChord.addLyric(
                            (chord_label_list[j]))
                    all_answers_per_chorale[j][
                        unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list[j]))] = \
                        all_answers_per_chorale[j].get(
                            unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list[j])),
                            0) + 1
                    sorted_result = sorted(all_answers_per_chorale[j].items(), key=lambda d: d[1], reverse=True)
                    # print(sorted_result[0])
                    ##print(sorted_result[0][-1])
                    # print(chord_tone_list[j])
                    for jj in range(len(sorted_result)):
                        if unify_GTChord_and_inferred_chord(translate_chord_name_into_music21(chord_label_list[j])) == \
                                sorted_result[jj][0]:  # when there are all different answers, choose the RB one!
                            break
                    if sorted_result[0][-1] == 1 or len(chord_tone_list[j]) <= 1:
                        chord = sorted_result[jj][0]  # output chords in the original key
                    else:
                        chord = sorted_result[0][0]
                    if j == 0:
                        thisChord.addLyric('Voting: ' + chord)
                    else:
                        thisChord.addLyric(chord)
                    if sorted_result[0] == sorted_result[-1]:  # Only one answer, agreed
                        thisChord.addLyric(' ')
                    else:
                        thisChord.addLyric('_!')
                    id_id = acc.findall(chord)
                    if id_id != []:  # has flat or sharp
                        root_ptr = re.search(r'[#-]+', chord).end()  # get the last flat or sharp position
                        transposed_result = transpose(chord[0: root_ptr], transposed_interval) + chord[root_ptr:]
                        sChords_new.recurse().getElementsByClass('Chord')[j].addLyric(transposed_result)
                        print(transposed_result, file=f_ori)
                        # print('original: ', line, 'transposed: ', transposed_result)
                    else:  # no flat or sharp, which means only the first element is the root
                        transposed_result = transpose(chord[0], transposed_interval) + chord[1:]
                        sChords_new.recurse().getElementsByClass('Chord')[j].addLyric(transposed_result)
                        print(transposed_result, file=f_ori)
                f_ori.close()
                f_transposed.close()
                s.write('musicxml',
                        fp=os.path.join(inputpath, 'predicted_result', fileName[i][
                                                                       :-4]) + '.xml')

                sNew.write('musicxml',
                           fp=os.path.join(inputpath, 'predicted_result', 'original_key', key_info, fn[
                                                                                                    :-4]) + '.xml')


if __name__ == "__main__":
    modelpath_NCT = os.path.join(os.getcwd(), 'ML_result', 'ISMIR2019',
                                 '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39',
                                 '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39_cv_1.hdf5')
    modelpath_CL = os.path.join(os.getcwd(), 'ML_result', 'ISMIR2019',
                                '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39',
                                '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39_cv_1_chord_tone.hdf5')
    modelpath_DH = os.path.join(os.getcwd(), 'ML_result', 'ISMIR2019',
                                '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39',
                                '3layer300DNNwindow_size1_2training_data1timestep0ISMIR2019NCT_pitch_classpitch_class3meter_NewOnset_New_annotation_keyC__training264_39_cv_1_direct_harmonic_analysis.hdf5')
    inputpath = os.path.join(os.getcwd(), 'new_muisc', 'New')
    predict_new_music(modelpath_NCT, modelpath_CL, modelpath_DH, inputpath)
    # inputpath = os.path.join(os.getcwd(), 'new_muisc', 'Praetorius')
    # predict_new_music(modelpath_NCT, modelpath_CL, inputpath)
    # inputpath = os.path.join(os.getcwd(), 'new_muisc', 'Schutz')
    # predict_new_music(modelpath_NCT, modelpath_CL, inputpath)
