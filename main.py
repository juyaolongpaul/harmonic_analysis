# This is the script to lead the user to train a neural network from scratch

from translate_output import annotation_to_txt
from transpose_to_C_chords import provide_path_12keys
from transpose_to_C_polyphony import transpose_polyphony
from get_input_and_output import generate_data_windowing_non_chord_tone_new_annotation_12keys
from predict_result_for_140 import train_and_predict_non_chord_tone
from chord_visualization import put_music21chord_into_musicXML
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Maximally melodic (modified version from Rameau) '
                             'or harmonic (music21 Chordify module) (default: %(default))',
                        type=str, default='melodic')
    parser.add_argument('-b', '--bootstrap',
                        help=' bootstrap the data (default: %(default)s)',
                        type=int, default=2)
    parser.add_argument('-a', '--augmentation',
                        help=' augment the data 12 times by transposing to 12 keys (default:%(default)',
                        type=str, default='Y')
    parser.add_argument('-l', '--num_of_hidden_layer',
                        help='number of units (default: %(default)s)',
                        type=int, default=2)
    parser.add_argument('-n', '--num_of_hidden_node',
                        help='number of units (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-m', '--model',
                        help='DNN, RNN and LSTM to choose from (default: %(default)s)',
                        type=str, default='DNN')
    parser.add_argument('-p', '--pitch',
                        help='use pitch or pitch class as input feature (default: %(default), the other option is pitch_class)',
                        type=str, default='pitch_class')
    parser.add_argument('-w', '--window',
                        help='the size of the input window (default: %(default))',
                        type=int, default=2)
    parser.add_argument('-pp', '--percentage',
                        help='the portion of the trainig data you want to use (a float number between 0-1'
                             ', not a percentage) (default: %(default))',
                        type=float, default=1)
    parser.add_argument('-c', '--cross_validation',
                        help='how many times do you want to cross validate (default: %(default))',
                        type=int, default=0)
    args = parser.parse_args()

    #print('This script will guide you to train a neural network (identify non-chord tones) from scratch')
    #print('The process contains 5 steps: (1) Get chord labels from my maximally melodic and '
         # 'harmonic annotation (2) Transpose the chorales and annotations to 12 keys '
          #'(3) Turning the chorales and annotations as training data (4) Train NN '
          #'(5) Visualization of the predicted non-chord tones in music score')

    annotation_to_txt()
    input = '.\\bach-371-chorales-master-kern\\kern\\' + 'chor'
    f1 = '.krn'
    output = '.\\genos-corpus\\answer-sheets\\bach-chorales\\New_annotation\\Melodic\\'
    f2 = '.txt'

    provide_path_12keys(input, f1, output, f2)
    transpose_polyphony()
    input = '.\\bach-371-chorales-master-kern\\kern\\'
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
    generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, input_dim, output_dim, 2,
                                                                 counter, counterMin, input, f1, output, f2, args.source,
                                                                 'Y', args.augmentation, args.pitch)
    generate_data_windowing_non_chord_tone_new_annotation_12keys(counter1, counter2, x, y, input_dim, output_dim, 2,
                                                                 counter, counterMin, input, f1, output, f2,
                                                                 args.source,
                                                                 'N', args.augmentation, args.pitch)
    train_and_predict_non_chord_tone(args.num_of_hidden_layer, args.num_of_hidden_node, args.window, args.percentage, args.model, 10, args.bootstrap, args.source, args.augmentation, args.cross_validation, args.pitch)
    #put_music21chord_into_musicXML('1')
if __name__ == "__main__":
    main()
