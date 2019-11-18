import xml.etree.cElementTree as ET
import os


def translate_FB_as_lyrics(number, suffix, prefix):
    """
    Translate FB in lists into string added to lyrics
    :param number:
    :param fix:
    :param duration:
    :return:
    """
    if prefix == 'sharp':
        prefix_sign = '#'
    elif prefix == 'flat':
        prefix_sign = 'b'
    elif prefix == 'natural':
        prefix_sign = prefix
    else:
        if prefix != '':
            print(prefix)
            input('this is the prefix you have not considered yet!')
        else:
            prefix_sign = prefix
    if suffix == 'backslash' or suffix == 'cross':
        suffix_sign = '#'
    elif suffix == 'natural':
        suffix_sign = suffix
    elif suffix == 'sharp':
        suffix_sign = '#'
    elif suffix == 'flat':
        suffix_sign = 'b'
    elif suffix == 'slash':
        suffix_sign = 'b'
    else:
        if suffix != '':
            print(suffix)
            input('this is the suffix you have not considered yet!')
        else:
            suffix_sign = suffix
    return prefix_sign+suffix_sign+number


def add_FB_content(j, each_fb, i, text):
    """
    Template to add figured bass content to lyrics
    :param j:
    :param each_fb:
    :param i:
    :return:
    """

    if i < len(each_fb['number']):
        if 'duration' in each_fb:
            if text.text is None:
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i]) + '+' + each_fb['duration']
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i]) + '+' + each_fb['duration']
        else:
            if text.text is None:
                text.text = translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i])
            else:
                text.text += translate_FB_as_lyrics(each_fb['number'][i], each_fb['suffix'][i], each_fb['prefix'][i])
        return text
    else:
        if j == 0: # this is when a figured bass does not have as much layers as the following ones
            text.text = ''
        return text

def add_FB_to_lyrics(note, fig):
    """
    Add figured bass info underneath as lyrics for the current bass note
    :param ele:
    :param fig:
    :return:
    """
    number_of_layers = 0
    for i, each_fb in enumerate(fig):
        if number_of_layers < len(each_fb['number']):
            number_of_layers = len(each_fb['number'])
    for i in range(number_of_layers):  # i means the current number of layer, horizontally
        print('numer of layers', i)
        lyric = ET.SubElement(note, 'lyric', {'number':str(i + 1)})  # create lyric sub element
        text = ET.SubElement(lyric, 'text')
        for j, each_fb in enumerate(fig):  # j means the ID of figures for the bass in the current layer
            if j == 0:
                text = add_FB_content(j, each_fb, i, text)
            else:
                text.text += ', '
                text = add_FB_content(j, each_fb, i, text)


def adding_XXXfix(each_FB_digit, name, single_XXXfix):
    XXXfix = each_FB_digit.find(name)
    if XXXfix is not None:
        single_XXXfix.append(each_FB_digit.find(name).text)
    else:
        single_XXXfix.append('')
    return single_XXXfix


if __name__ == '__main__':
    for filename in os.listdir(os.path.join('.', '371_FB')):
        if 'FB.musicxml' not in filename: continue
        #if '017' not in filename: continue
        print(filename)
        tree = ET.ElementTree(file=os.path.join('.', '371_FB', filename))
        for elem in tree.iter(tag='part'):  # get the bass voice
            if elem.attrib['id'] == 'P4':
                child = elem
        for measure in child.iter(tag='measure'):  # get all the measures within the bass voice
            print(measure.attrib)
            fig = []  # empty FB dictionary for each bass note
            for ele in measure.iter():
                if ele.tag == 'figured-bass' or ele.tag == 'note':
                    #print(list(ele))
                    if ele.tag == 'figured-bass':  # found a (compound) FB label
                        FB_digit = {}
                        single_digit = []  # list for all the digits
                        single_suffix = []  # list for all the suffix
                        single_prefix = []
                        for each_FB_digit in ele.iter():
                            if each_FB_digit.tag == 'figure':
                                single_digit = adding_XXXfix(each_FB_digit, 'figure-number', single_digit)
                                single_suffix = adding_XXXfix(each_FB_digit, 'suffix', single_suffix)
                                single_prefix = adding_XXXfix(each_FB_digit, 'prefix', single_prefix)
                                FB_digit['number'] = single_digit
                                FB_digit['suffix'] = single_suffix
                                FB_digit['prefix'] = single_prefix
                            if each_FB_digit.tag == 'duration':
                                FB_digit['duration'] = each_FB_digit.text


                        fig.append(FB_digit)
                        print(fig)
                    if ele.tag == 'note':
                        if fig != [] and fig != [{}] and 'number':
                            print("add the FB to lyrics")
                            # if fig[0]['number'] == ['8', '4']:
                            #     print('debug')
                            add_FB_to_lyrics(ele, fig)
                            fig = []  # reset the FB for the next note with FB
        tree.write(open(os.path.join('.', '371_FB', filename[:-9] + '_' + 'lyric' + '.xml'), 'w'), encoding='unicode')
        # till this point, all FB has been extracted and attached as lyrics underneath the bass line!