import os
import re
replace = '-*![](){}\n'
dic = {}
for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
    if(file_name[:5] == 'trans'):
        f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
        print(file_name)
        for line in f.readlines():
            '''for i, letter in enumerate(line):
                if(letter not in ' ¸-#+°/[](){}\n'):
                    if(letter.isalpha() == 0 and letter.isdigit() == 0):

                        print('special' + letter)
                        print(line)'''
            for letter in replace:
                line = line.replace(letter, '')
            line = re.sub(r'/\w+', '', line)
            print(line)
            for chord in line.split():
                dic.setdefault(chord, 0)
                dic[chord] += 1
li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
fchord = open('chord_frequency_no_inversion.txt', 'w')
total_freq = 0
total_percentage = 0
for word in li:
    total_freq += word[1]
for word in li:
    print (word, end = '', file = fchord)
    total_percentage += word[1] / total_freq
    print(str(word[1] / total_freq), end = '', file = fchord)
    print('total: ' + str(total_percentage), file = fchord)


