f = open('midi.bat','w')
for i in range(1,372):
    if i<10:
        print('lilypond C:\\Users\\User\\PycharmProjects\\harmonic_analysis\\bach-chorales\\00' + str(i) + '.ly', file = f)
    elif i<100:
        print('lilypond C:\\Users\\User\\PycharmProjects\\harmonic_analysis\\bach-chorales\\0' + str(i) + '.ly' , file = f)
    else:
        print('lilypond C:\\Users\\User\\PycharmProjects\\harmonic_analysis\\bach-chorales\\' + str(i) + '.ly', file = f)
f.close()

