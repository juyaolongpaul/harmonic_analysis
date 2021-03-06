# This is the script to lead the user to train a neural network from scratch

import argparse
import os

from get_input_and_output import generate_data_windowing_non_chord_tone_new_annotation_12keys
from kernscore import extract_chord_labels
from predict_result_for_140 import train_and_predict_non_chord_tone
from translate_output import annotation_translation
from transpose_to_C_chords import provide_path_12keys
from transpose_to_C_polyphony import transpose_polyphony


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Maximally melodic (modified version from Rameau) '
                             'or rule_MaxMel (default: %(default)) or Rameau',
                        type=str, default='ISMIR2019')
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
                        type=str, default='pitch_class_with_bass')
    parser.add_argument('-w', '--window',
                        help='the size of the input window (default: %(default))',
                        type=int, default=2)
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
    args = parser.parse_args()
    if args.source == 'Rameau':
        input = os.path.join('.', 'bach_chorales_scores', 'original_midi+PDF')
        f1 = '.mid'
    else:
        input = os.path.join('.', 'bach-371-chorales-master-kern', 'kern')
        f1 = '.krn'  # the version of chorales used
    output = os.path.join('.', 'genos-corpus', 'answer-sheets', 'bach-chorales', 'New_annotation', args.source)
    f2 = '.txt'
    if args.source.find('rule_MaxMel') == -1:
        extract_chord_labels(output, f1)
    annotation_translation(input, output, args.version, args.source)  # A function that extract chord labels from musicxml to txt and translate them
    provide_path_12keys(input, f1, output, f2, args.source)  # Transpose the annotations into 12 keys
    transpose_polyphony(args.source, input)  # Transpose the chorales into 12 keys
    if args.source != 'Rameau':
        f1 = '.xml'
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
    generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, input_dim, output_dim, args.window,
                                                                 counter, counterMin, input, f1, output, f2,
                                                                 args.source,
                                                                 args.augmentation, args.pitch, args.ratio,
                                                                 args.cross_validation, args.version, args.output, args.input)  # generate training and testing data, return the sequence of test id
      # only execute this when the CV matrices are complete

    # train_and_predict_non_chord_tone(args.num_of_hidden_layer, args.num_of_hidden_node, args.window, args.percentage,
    #                                  args.model, args.timestep, args.bootstrap, args.source, args.augmentation,
    #                                  args.cross_validation, args.pitch, args.ratio, input, output, args.balanced, args.output, args.input, args.predict)
    train_and_predict_non_chord_tone(args.num_of_hidden_layer, args.num_of_hidden_node, args.window, args.percentage,
                                     args.model, args.timestep, args.bootstrap, args.source, args.augmentation,
                                     args.cross_validation, args.pitch, args.ratio, input, output, args.balanced,
                                     args.output, args.input, args.predict, ['099', '193', '210', '345', '053', '071', '104',
                                                                             '133', '182', '227', '232', '238', '243', '245', '259'
        , '261', '271', '294', '346', '239', '282', '080',
                               '121', '136', '137', '139', '141', '156', '179', '201', '247', '260', '272', '275',
                               '278', '289', '308', '333', '365']) # Evaluate on the 39 reserved chorales
    #put_non_chord_tone_into_musicXML(input, output, args.source, f1, f2, args.pitch)  # visualize as scores
if __name__ == "__main__":
    main()
