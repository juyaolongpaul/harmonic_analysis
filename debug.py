from music21 import *
'''import os
for file_name in os.listdir('.\\bach_chorales_scores\\transposed_MIDI\\'):
    print(file_name)
    s = converter.parse('.\\bach_chorales_scores\\transposed_MIDI\\' + file_name)
    k = s.analyze('key')
    print(k.name)
    if(k.name != 'C major' and k.name != 'A minor'):
        print('error:' + k.name)
'''
s = converter.parse(r"F:\我的坚果云\Temp\Harmonic_analysis\annotation003_Yaolong.xml")
#s = converter.parse(r"F:\PyCharmProjects\harmonic_analysis\predicted_result\115.xml")
sChords = s.chordify()
#l = text.assembleAllLyrics(s)
#print(l)
#sChords.show()
#s.show()
sChords = s.parts[4]
#bass.show()
#for n in .recurse().getElementsByClass('Note'):
for i, thisChord in enumerate(sChords.recurse().getElementsByClass('Chord')):

    print(thisChord.lyrics[3].text)