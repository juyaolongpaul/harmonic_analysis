import os
import re
dic = {}


def get_chord_line(line, sign):
    """

    :param line:
    :param replace:
    :return:
    """
    for letter in '!':
        line = line.replace(letter, '')
    if(sign == '0'):
        line = re.sub(r'/\w[b#]*', '', line)  # remove inversions
    return line


def calculate_freq(dic, line):
    """
    :param dic:
    :param line:
    :return:
    """
    for chord in line.split():
        dic.setdefault(chord, 0)
        dic[chord] += 1
    return dic


def output_freq_to_file(filename, dic):
    """

    :param filename:
    :param dic:
    :return:
    """
    li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    fchord = open(filename, 'w')
    total_freq = 0
    total_percentage = 0
    for word in li:
        total_freq += word[1]
    for word in li:
        print(word, end='', file=fchord)
        total_percentage += word[1] / total_freq
        print(str(word[1] / total_freq), end='', file=fchord)
        print(' total: ' + str(total_percentage), file=fchord)
if __name__ == "__main__":
    sign = input("do you want inversions or not? 1: yes, 0: no")

    for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
        if(file_name[:6] == 'transl'):
            f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
            print(file_name)
            for line in f.readlines():
                '''for i, letter in enumerate(line):
                    if(letter not in ' ¸-#+°/[](){}\n'):
                        if(letter.isalpha() == 0 and letter.isdigit() == 0):

                            print('special' + letter)
                            print(line)'''
                line = get_chord_line(line, sign)
                print(line)
                dic = calculate_freq(dic, line)
    if(sign == '1'):
        output_freq_to_file('chord_frequency.txt', dic)
    else: output_freq_to_file('chord_frequency_no_inversion.txt', dic)


