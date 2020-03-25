# This is the script to lead the user to train a neural network from scratch

import argparse
import os

from get_input_and_output import generate_data_windowing_non_chord_tone_new_annotation_12keys_FB
from kernscore import extract_chord_labels
from predict_result_for_140 import train_and_predict_FB
from translate_output import annotation_translation
from transpose_to_C_chords import provide_path_12keys
from transpose_to_C_polyphony import transpose_polyphony_FB
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='which FB source you wanna use (default: %(default))',
                        type=str, default='Bach_o_FB')
    parser.add_argument('-b', '--bootstrap',
                        help=' bootstrap the data (default: %(default)s)',
                        type=int, default=0)
    parser.add_argument('-a', '--augmentation',
                        help=' augment the data 12 times by transposing to 12 keys (default:%(default)',
                        type=str, default='Y')
    parser.add_argument('-l', '--num_of_hidden_layer',
                        help='number of units (at least two layers) (default: %(default)s)',
                        type=int, default=3)
    parser.add_argument('-n', '--num_of_hidden_node',
                        help='number of units (default: %(default)s)',
                        type=int, default=300)
    parser.add_argument('-m', '--model',
                        help='DNN, RNN and LSTM to choose from (default: %(default)s)',
                        type=str, default='DNN')
    parser.add_argument('-p', '--pitch',
                        help='use pitch or pitch class or pitch class binary or pitch class 4 voices as '
                             'input feature. You can also append 7 in the end to use '
                             'do the generic pitch(default: %(default)',
                        type=str, default='pitch_class_with_bass_scale_new_data')
    parser.add_argument('-w', '--window',
                        help='the size of the input window (default: %(default))',
                        type=int, default=1)
    parser.add_argument('-pp', '--percentage',
                        help='the portion of the training data you want to use (a float number between 0-1'
                             ', not a percentage) (default: %(default))',
                        type=float, default=1)
    parser.add_argument('-c', '--cross_validation',
                        help='how many times do you want to cross validate (default: %(default))',
                        type=int, default=10)
    parser.add_argument('-r', '--ratio',
                        help='the portion of the trainig data you want to use (a float number between 0-1'
                             ', not a percentage. 0.6 means 60% for training, 40% for testing) (default: %(default))',
                        type=float, default=0.8)
    parser.add_argument('-v', '--version',
                        help='whether to use 153 chorales (same with Rameau) or 367 chorales (rule-based) (default: %(default))',
                        type=int, default=367)
    parser.add_argument('-bal', '--balanced',
                        help='specify whether you want to make the dataset balanced (default: %(default))',
                        type=int, default=0)
    parser.add_argument('-o', '--output',
                        help='specify whether you want output non-chord tone (NCT or NCT_pitch_class) or chord labels (CL) directly (default: %(default))',
                        type=str, default='NCT_pitch_class')
    parser.add_argument('-i', '--input',
                        help='specify what input features, besides pitch, you are using (default: %(default))',
                        type=str, default='3meter_NewOnset')
    parser.add_argument('-time', '--timestep',
                        help='specify how many time steps (default: %(default))',
                        type=int, default=0)
    parser.add_argument('-pre', '--predict',
                        help='specify whether you want to predict and output the result in XML (default: %(default))',
                        type=str, default='Y')
    parser.add_argument('-ru', '--rule',
                        help='specify which rules you wanna use (default: %(default))',
                        type=list, default=['NCT bass', 'NCT upper voices', 'FB already labeled', '16th (or shorter) note slice ignored'])
    args = parser.parse_args()
    f1 = '.xml'
    f2 = '.txt'
    if args.source == 'Bach_o_FB':
        input = os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'musicXML_master')
    elif args.source == 'Bach_e_FB':
        input = os.path.join('.', 'Bach_chorale_FB', 'FB_source', 'musicXML_master', 'editorial_FB_only')
    # transpose_polyphony_FB(args.source, input)  # Transpose the chorales into 12 keys
    # if args.source != 'Rameau':
    #     f1 = '.xml'
    counter1 = 0  # record the number of salami slices of poly
    counter2 = 0  # record the number of salami slices of chords
    counter = 0
    counterMin = 60
    # Get input features
    sign = '0'  # input("do you want inversions or not? 1: yes, 0: no")
    output_dim = '12'  # input('how many kinds of chords do you want to calculate?')
    window_size = '0'  # int(input('how big window?'))
    output_dim = int(output_dim)
    input_dim = 12
    x = []
    y = []
    generate_data_windowing_non_chord_tone_new_annotation_12keys_FB(counter1, counter2, x, y, input_dim, output_dim, args.window,
                                                                 counter, counterMin, input, f1, input, f2,
                                                                 args.source,
                                                                 args.augmentation, args.pitch, args.ratio,
                                                                 args.cross_validation, args.version, args.output, args.input)  # generate training and testing data, return the sequence of test id
    #   # only execute this when the CV matrices are complete
    #
    # # train_and_predict_non_chord_tone(args.num_of_hidden_layer, args.num_of_hidden_node, args.window, args.percentage,
    # #                                  args.model, args.timestep, args.bootstrap, args.source, args.augmentation,
    # #                                  args.cross_validation, args.pitch, args.ratio, input, output, args.balanced, args.output, args.input, args.predict)
    train_and_predict_FB(['NCT bass', 'NCT upper voices', 'FB already labeled', '16th (or shorter) note slice ignored'], args.num_of_hidden_layer, args.num_of_hidden_node, args.window, args.percentage,
                                     args.model, args.timestep, args.bootstrap, args.source, args.augmentation,
                                     args.cross_validation, args.pitch, args.ratio, input, input, args.balanced,
                                     args.output, args.input, args.predict, ['8.06', '161.06a', '161.06b', '16.06', '48.07', '195.06', '149.07']) # Evaluate on the reserved chorales, where the 4th ones and onward are the ones missing FB a lot
    # this gives us 124 training chorales: 143-12 interlude chorales - 7 chorales with both missing figures and the first three where music21 has issues. Look into 297!
    #put_non_chord_tone_into_musicXML(input, output, args.source, f1, f2, args.pitch)  # visualize as scores
if __name__ == "__main__":
    main()
