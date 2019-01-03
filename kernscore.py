"""
Stuff for reading in Humdrum files.
Only handles basic functionality right now.
"""
import re
import os
from midiutil.MidiFile import MIDIFile

PITCHES_RE = re.compile('[ra-gA-Gn#\-]+')
RECIP_RE = re.compile('[0-9.]+')
MODIFIERS_RE = re.compile('[^ra-gA-Gn#\-0-9.]')


class KernScore(object):
    """Python class representing a .krn score.
    Only verified for Bach Chorales right now.
    Instantiate using KernScore(path_to_kernfile).
    """
    def __init__(self):
        self.file_path = None

        self.metadata = {}
        self.comments = []
        self.section_order = []

        self.sections = []
        self.barlines = []
        self.parts = []

    @property
    def cadences(self):
        """Chorale cadences, as judged by fermata location.
        """
        part_fermatas = []
        for part in self.parts:
            part_fermatas.append([ event for event in part['events']
                                   if ';' in event['modifiers'] ])

        cadences = []
        for stack in zip(*part_fermatas):
            cadences.append(new_cadence(stack))

        return cadences

    def import_kernfile(self, file_path):
        """Import a kernfile and overwrite the internal state of the KernScore.
        """
        if self.file_path:
            self.__init__()

        self.file_path = file_path

        # Partwise markers.
        next_beats = []

        kernfile = open(file_path)
        for line in kernfile:
            line = line.strip()

            # Parse comments.
            if line.startswith('!!!'):
                refkey = line[3:6]
                self.metadata[refkey] = line[8:]

            elif line.startswith('!!'):
                self.comments.append(line[4:])

            elif line.startswith('!'):
                # Discard inline comments.
                pass

            # Parse scorewide interpretations.
            elif '*>[' in line:
                self.section_order = line[3:-1].split(',')

            elif '*>' in line:
                self.sections.append(new_section(line, min(next_beats)))
                
            elif '*-' in line:
                # That's all, folks.
                pass

            # Parse spinewise interpretations.
            elif line.startswith('*'):
                sample_line = line.split('\t')
                number_of_parts = len(sample_line)
                for i, token in enumerate(line.split('\t')):
                    if token == '**kern':
                        # Create a new part, and initialize an entry
                        # in the next_beats array.
                        self.parts.append(new_part(token))
                        next_beats.append(0)
                    elif token == '**chordsymbol':
                        self.parts.append(new_part(token))
                        next_beats.append(0)
                        self.parts[i]['instrument_class'] = token
                    elif token.startswith('*IC'):
                        self.parts[i]['instrument_class'] = token.lstrip('*IC')

                    elif token.startswith('*I'):
                        self.parts[i]['instrument'] = token.lstrip('*I')

                    elif token.startswith('*k'):
                        self.parts[i]['key_sig'] = token.lstrip('*k')

                    elif token.startswith('*M'):
                        self.parts[i]['time_sig'] = token.lstrip('*M')

                    elif token.startswith('*clef'):
                        self.parts[i]['clef'] = token.lstrip('*clef')


            # Parse data tokens.
            elif '=' in line:
                self.barlines.append(new_barline(line, min(next_beats)))

            else:
                tokens = [ new_token(string, next_beats[i], i, number_of_parts)
                           for i, string in enumerate(line.split('\t')) ]

                for i, token in enumerate(tokens):
                    token and self.parts[i]['events'].append(token)
                    next_beats[i] += token.get('duration', 0)
        f = open(file_path[:-3]+'txt','w')
        for i, item in enumerate(self.parts[number_of_parts - 1]['events']):
            if len(item['chord'])>1:
                print(item['chord'], file=f)
            else:
                print(item['chord'])
        kernfile.close()

    def export_midi(self, file_path):
        """Export a MIDI file."""
        midi = MIDIFile(1)
        midi.addTrackName(0, 0, self.metadata.get('OTL'))
        midi.addTempo(0, 0, 80)

        for i, part in enumerate(self.parts):
            non_rests = [ d for d in part['events'] if d['pitch'] != 'r' ]
            for note in non_rests:
                midi.addNote(track=0, channel=i,
                             pitch=note['midinote'],
                             time=note['beat'],
                             duration=note['duration'],
                             volume=80)

        with open(file_path, 'wb') as binfile:
            midi.writeFile(binfile)


# Sub-parsers / models.
def new_part(declaration):
    return { 'declaration': declaration,
             'events': [] }


def new_barline(kern_line, beat):
    """Make a new barline dict.
    """
    barline = {'beat': beat}
    first_token = kern_line.split('\t')[0]

    if '==@' in first_token:
        barline = { 'type': 'final',
                    'number': None }
    elif '==' in first_token:
        barline = { 'type': 'double',
                    'number': first_token[2:] }
    elif '=' in first_token:
        barline = { 'type': 'single',
                    'number': first_token[1:] }

    return barline


def new_section(kern_line, beat):
    """Make a new section dict.
    """
    first_token = kern_line.split('\t')[0]

    return {
        'beat': beat,
        'section': first_token[2:],
    }


def new_token(token_string, beat, i, number_of_parts, timebase=4):
    """Create a new token dictionary from a kern token string.

    @param token_string: A single humdrum token.
    @param beat: The beat the token falls on.
    @param timebase: The recip indication of the beat note.
    @return: a token dict.
    """
    if token_string[0] == '.':
        token = {}

    else:
        pitch = ''.join(PITCHES_RE.findall(token_string))
        modifiers = ''.join(MODIFIERS_RE.findall(token_string))
        recip = ''.join(RECIP_RE.findall(token_string))
        if i == number_of_parts - 1:
            token = {
                'chord': token_string
            }
        else:
            token = {
                'pitch': pitch,
                'recip': recip,
                'beat': beat,
                'modifiers': modifiers,
            }

    return token


def new_cadence(partstack):
    """Create a new cadence object using a partstack.
       TODO: figure out what really should be represented.
    """
    return {
        'pitches': tuple( p['pitch'] for p in partstack ),
        'midinotes': tuple( p['midinote'] for p in partstack ),
        'beats': tuple( p['beat'] for p in partstack ),
    }


def extract_chord_labels(file_path, filetype):
    """
    Extract chord labels from the kern file generated by rule-based model
    :param file_path:
    :param source:
    :return:
    """
    print('Step 1: Extract chord labels from the Kern files')
    x = KernScore()
    for file_name in os.listdir(file_path):
        if file_name[-4:] == filetype:
            print(file_name)
            x.import_kernfile(os.path.join(file_path,file_name))