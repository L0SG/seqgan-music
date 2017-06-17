from music21 import *
import os
from copy import copy
import pickle


def load_data(file_path):
    ## load midi file using music21 library
    piece = converter.parse(file_path)
    """
    # transpose all streams to C major. this process is to reduce the number of states
    # store the key of music before transposition.
    k = pre_piece.analyze('key')
    # save the interval of C and current key
    if k.mode == 'minor':
        i = interval.Interval(k.parallel.tonic, pitch.Pitch('C'))
    else:
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
    # transpose the music using stored interval
    piece = pre_piece.transpose(i)
    # return transposed music
    """
    return piece


class preprocessing(object):
    def __init__(self):
        ## to collect and count unique chords, notes, octaves
        # lists that store unique chords and octaves
        self.chords = []
        self.chord_octaves = []
        # lists for counting the number of times the chords and octaves appear
        self.chords_cnt = [0] * len(self.chord_ref)
        self.chord_octaves_cnt = [0] * len(self.octave_ref)
        # the same thing about notes
        self.notes = []
        self.note_octaves = []
        self.notes_cnt = [0] * len(self.note_ref)
        self.note_octaves_cnt = [0] * len(self.note_octave_ref)

    def parsing(self, data_path):
        # load midi file
        piece = load_data(data_path)
        # all_parts is list of melody and chords, each is sequence of [start time, duration, octave, pitch, velocity]
        all_parts = []
        # for all parts in midi file (most of our data have two parts, melody and chords)
        for part in piece.iter.activeElementList:
            """ # check that the part is a piano song.
            # save the instrument name.
            try:
                track_name = part[0].bestName()
            except AttributeError:
                track_name = 'None'
            part_tuples.append(track_name)

            """
            # part_tuples is sequence of [start time, duration, octave, pitch]
            part_tuples = []
            for event in part._elements:
                # if chords or notes exist recursive (sometimes this happened in a complicated piano song file)
                if event.isStream:
                    _part_tuples = []
                    for i in event._elements:
                        _part_tuples = self.streaming(event, i, _part_tuples)
                    all_parts.append(_part_tuples)
                # normal case
                else:
                    part_tuples = self.streaming(part, event, part_tuples)
            if part_tuples != []:
                all_parts.append(part_tuples)
        return all_parts

    def streaming(self, part, event, part_tuples):
        # save start time
        for y in event.contextSites():
            if y[0] is part:
                offset = y[1]
        # if current event is chord
        if getattr(event, 'isChord', None) and event.isChord:
            # chord pitch wjdfl
            octaves = []
            for pitch in event.pitches:
                octaves.append(pitch.octave)
            sort_idx = [i[0] for i in sorted(enumerate(event.pitchClasses), key=lambda x: x[1])]
            octaves = [x for (y, x) in sorted(zip(sort_idx, octaves))]

            if event.orderedPitchClasses not in self.chords:
                self.chords.append(event.orderedPitchClasses)
            if octaves not in self.chord_octaves:
                self.chord_octaves.append(octaves)

        # if current event is note
        if getattr(event, 'isNote', None) and event.isNote:
            # change to key
            # make one step in sequence

            if event.pitch.octave not in self.note_octaves:
                self.note_octaves.append(event.pitch.octave)
            if event.pitchClass not in self.notes:
                self.notes.append(event.pitchClass)

        # if current event is rest
        if getattr(event, 'isRest', None) and event.isRest:
            part_tuples.append([offset, event.quarterLength, 0, 0, 0])
        return part_tuples

    def compare_parts(self, all_parts):
        if len(all_parts) < 2:
            raise ValueError('the number of parts is less than two!')
        melody = all_parts[0]
        chord = all_parts[1]
        while 1:
            for i in range(len(melody)):
                try:
                    if melody[i][0] < chord[i][0]:
                        chord.insert(i, [melody[i][0], 0, 0, 0, 0])
                except:
                    chord.append([melody[i][0], 0, 0, 0, 0])
            if self.chk_same(melody, chord):
                return all_parts

            for i in range(len(chord)):
                try:
                    if chord[i][0] < melody[i][0]:
                        melody.insert(i, [chord[i][0], 0, 0, 0, 0])
                except:
                    # if length of chord is bigger than that of melody
                    melody.append([chord[i][0], 0, 0, 0, 0])
            if self.chk_same(melody, chord):
                return all_parts

    def chk_same(self, melody, chord):
        mel_time = [item[0] for item in melody]
        cho_time = [item[0] for item in chord]
        if mel_time == cho_time:
            return True
        else:
            return False

    def sequentialize(self, parsed):
        if len(parsed[0]) != len(parsed[1]):
            raise ValueError
        sequence = []
        for i in range(len(parsed[0])):
            token = copy(parsed[0][i][1:])
            token.extend(parsed[1][i][1:])
            sequence.append(token)
        return sequence


if __name__ == "__main__":

    # preprocessing
    a = preprocessing()
    data_dir = './Nottingham/all/'
    dataset = []
    for file in os.listdir(data_dir):
        print(file)
        seq = a.parsing(data_dir + file)
        print('notes: ', a.notes)
        print('len(notes): ', len(a.notes))
        print('note_octaves: ', a.note_octaves)
        print('len(note_octaves): ', len(a.note_octaves))
        print('chords: ', a.chords)
        print('len(chords): ', len(a.chords))
        print('chord_octaves: ', a.chord_octaves)
        print('len(chord_octaves): ', len(a.chord_octaves))
        print('\n')
