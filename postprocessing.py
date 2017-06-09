from music21 import *
import pickle

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
    if token[1]==0:
        r = note.Rest()
        r.duration.quarterLength = token[0]
        event = midi.translate.noteToMidiEvents(r)
    # if token is note
    elif 0 < token[1] < 7:
        p = convert_pitch(token)
        n = note.Note(p)
        n.volume.velocity = token[3]
        n.duration.quarterLength = token[0]
        event = midi.translate.noteToMidiEvents(n)
    # if token is chord
    else:
        p = convert_pitch(token)
        c = chord.Chord(p)
        c.volume.velocity = token[3]
        c.duration.quarterLength = token[0]
        event = midi.translate.chordToMidiEvents(c)
    return event

# convert (octave and key) to pitch for midi file
def convert_pitch(token):
    octave_ind = token[1]
    key_ind = token[2]
    # list of scale (C)
    scale = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    # find octave and key
    octave = octaves_ref[octave_ind]
    key = chords_ref[key_ind]
    # check the number of key in chord is same in octave
    assert len(octave) == len(key)
    # convert octave and key to pitch string
    p = []
    for i in xrange(len(key)):
        p.append(scale[key[i]]+str(octave[i]))
    return p

def main():
    # assumption : data is one sequence list, len(seq)=20, element of seq is integer
    data = [209,191,502,117,503,7,492,9,152,5,438,278,331,39,35,508,140,509,9,106]

    sequence = inverse_mapping(data)
    melody, chords = split(sequence)

    #fp = stream.write('midi', fp='./dataset/')


    mt = midi.MidiTrack(1)
    t=0
    tLast=0
    for token in data:
        token




        dt = midi.DeltaTime(mt)
        dt.time = t-tLast
        #add to track events
        mt.events.append(dt)

        me=midi.MidiEvent(mt)
        me.type="NOTE_ON"
        me.channel=1
        me.time= None #d
        me.pitch = p
        me.velocity = v
        mt.events.append(me)

        # add note off / velocity zero message
        dt = midi.DeltaTime(mt)
        dt.time = d
        # add to track events
        mt.events.append(dt)

        me=midi.MidiEvent(mt)
        me.type="NOTE_ON"
        me.channel=1
        me.time= None #d
        me.pitch = p
        me.velocity = 0
        mt.events.append(me)

if __name__ == "__main__":
    main()