from get_input_and_output import get_chord_line
from get_input_and_output import calculate_freq
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np
#sign = input("do you want inversions or not? 1: yes, 0: no")
#output_dim =  input('how many kinds of chords do you want to calculate?')
#window_size = int(input('how big window?'))
'''sign = 0
output_dim = 30
window_size = 1'''
output_dim = 50
input_dim = 12
dic = {}
for file_name in os.listdir('.\\genos-corpus\\answer-sheets\\bach-chorales'):
    if (file_name[:6] == 'transl'):
        f = open('.\\genos-corpus\\answer-sheets\\bach-chorales\\' + file_name, 'r')
        print(file_name)
        for line in f.readlines():
            '''for i, letter in enumerate(line):
                if(letter not in ' ¸-#+°/[](){}\n'):
                    if(letter.isalpha() == 0 and letter.isdigit() == 0):

                        print('special' + letter)
                        print(line)'''
            line = get_chord_line(line, '0')
            print(line)
            dic = calculate_freq(dic, line)
li = sorted(dic.items(), key=lambda d: d[1], reverse=True)
total_freq = 0

for word in li:
    total_freq += word[1]
list_of_chords = []
list_of_freq = []
for i, word in enumerate(li):
    if(i == output_dim - 1):  # the last one is 'others'
        break
    list_of_chords.append(word[0])
    list_of_freq.append(word[1]/total_freq)
print (list_of_chords)  # Get the top 35 chord freq
print (list_of_freq)
print(list_of_chords)
print(list_of_freq)
y_pos = np.arange(len(list_of_chords))
plt.bar(y_pos, list_of_freq)
plt.xticks(y_pos, list_of_chords)

plt.show()