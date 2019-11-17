from music21 import *
import xml.etree.cElementTree as ET

tree = ET.ElementTree(file=r'D:\Nutstore\Temp\Harmonic_analysis\FB\013_FB_encoded_MuseScore_figure-bass_no_underline_part.musicxml')
for elem in tree.iter(tag='part'):  # get the bass voice
    if elem.attrib['id'] == 'P4':
        child = elem
for measure in child.iter(tag='measure'):  # get all the measures within the bass voice
    print(measure.attrib)
    for ele in measure.iter():
        if ele.tag == 'figured-bass' or ele.tag == 'note':
            print(list(ele))
