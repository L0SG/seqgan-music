from music21 import *
import pickle
from random import randint

with open('./dataset/chords', 'rb') as fp:
    chords_ref = pickle.load(fp)
with open('./dataset/octaves', 'rb') as fp:
    octaves_ref = pickle.load(fp)

# map sequence of token (integer) to sequence of [melody duration, octave, key, velocity, chord duration, octave, key, velocity]
def inverse_mapping(tokens):
    output = []
    # load reference list
    with open('./dataset/tokens', 'rb') as fp:
        tokens_ref = pickle.load(fp)
    # inverse mapping : token is index of tokens_ref
    for token in tokens:
        output.append(tokens_ref[token])
    return output
# now the length of sequence is still 20, and length of each element is 8

# split melody and chords
def split(seq):
    melody = []
    chords = []
    for token in seq:
        # in every token, first four elements are melody and last four elements are chord
        melody.append(token[0:4])
        chords.append(token[4:])
    return melody, chords

# make note from given token [duration, octave, key, velocity]
def make_event(token):
    # if token is rest
    if token[2]==0:
        r = note.Rest()
        r.duration.quarterLength = token[0]
        event = r# midi.translate.noteToMidiEvents(r)
    # if token is note
    elif 0 < token[2] < 13:
        p = convert_pitch(token)
        n = note.Note(p[0])
        n.volume.velocity = token[3]
        n.duration.quarterLength = token[0]
        event = n#midi.translate.noteToMidiEvents(n)
    # if token is chord
    else:
        p = convert_pitch(token)
        c = chord.Chord(p)
        c.volume.velocity = token[3]
        c.duration.quarterLength = token[0]
        event = c#midi.translate.chordToMidiEvents(c)
    return event

# convert (octave and key) to pitch for midi file
def convert_pitch(token):
    # change elements of list from float to integer
    octave_ind = int(token[1])
    key_ind = int(token[2])
    # list of scale (C)
    scale = ['C#','D','D#','E','F','F#','G','G#','A','A#','B','C']
    # find octave and key
    octave = octaves_ref[octave_ind]
    key = chords_ref[key_ind]
    # check the number of key in chord is same in octave
    assert len(octave) == len(key)
    # convert
    # convert octave and key to pitch string
    p = []
    for i in xrange(len(key)):
        p.append(scale[key[i]]+str(octave[i]))
    return p

def main(num_sample):
    # load sequence file
    with open('./dataset/train', 'rb') as fp:
        seq = pickle.load(fp)

    for sample in xrange(num_sample):
        # select random sample of sequence
        seq_idx = randint(0,len(seq))
        data = seq[0]
        # assumption : data is one sequence list, len(seq)=100, element of seq is integer

        #real_data = [209,191,502,117,503,7,492,9,152,5,438,278,331,39,35,508,140,509,9,106]
        #data = [39, 652, 80, 747, 61, 36, 3285, 2495, 136, 117, 208, 4, 251, 38, 4, 40, 38, 4, 76, 4]

        sequence = inverse_mapping(data)
        melody, chords = split(sequence)



        all_parts = stream.Stream()

        # make melody stream
        part_melody = stream.Part()
        for token in melody:
            # skip dummy rest
            if token != [0, 0, 0, 0]:
                event = make_event(token)
                # append event to part of melody
                part_melody.append(event)

        # make chord stream
        part_chord = stream.Part()
        chk_first = 1
        offset = 0
        for i in xrange(len(chords)):
            # skip dummy rest
            if chords[i] != [0, 0, 0, 0]:
                # match fist start time of chord
                if chk_first == 1:
                    offset = part_melody[i].offset
                    chk_first = 0
                event = make_event(chords[i])
                # append event to part of chord
                part_chord.append(event)
                part_chord[-1].offset += offset


        all_parts.append([part_melody, part_chord])
        fp = all_parts.write('midi', './midi/test_' + str(sample) +'.mid')
        print('file name:',fp)


if __name__ == "__main__":
    main(1)