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
        # dictionaries of (notes and chords) and (octaves of notes and octaves of chords)
        self.chord_ref = ['Rest', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, [2, 7, 11], [2, 6, 9], [0, 5, 9], [0, 4, 7], [2, 5, 10], [4, 7, 11], [1, 4, 9], [2, 6, 11], [0, 4, 9], [4, 8, 11], [2, 4, 8, 11], [2, 5, 9], [2, 7, 10], [3, 6, 11], [1, 6, 9], [1, 4, 7, 9], [1, 6, 10], [2, 5, 7, 11], [0, 2, 6, 9], [2, 6], [1, 4, 7], [0, 4, 7, 10], [3, 6, 9, 11], [1, 4, 8], [1, 4, 6, 10], [0, 3, 7], [2, 4, 7, 11], [0, 4], [2, 5, 11], [0, 3, 5, 9], [3, 7, 10], [2, 5, 8, 10], [0, 5, 8], [0, 4, 7, 9], [0, 3, 8]]
        self.octave_ref = ['Rest', 2, 3, 4, 5, 6, [3, 4, 3], [3, 3, 3], [4, 3, 4], [3, 3, 4, 3], [4, 3, 4, 4], [4, 4, 3, 3], [3, 3], [4, 4, 4], [3, 3, 3, 3], [3, 4, 4], [4, 4, 3], [4, 3, 3], [4, 4, 5], [4, 5, 4, 5], [4, 4, 4, 3]]
        self.note_ref = ['Rest', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.note_octave_ref = ['Rest', 2, 3, 4, 5, 6]
        """ # these are not necessary for creating dataset.
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
        ]
        self.notes_cnt = [0] * len(self.note_ref)
        self.note_octaves_cnt = [0] * len(self.note_octave_ref)
        """

    def parsing(self, data_path):
        # load midi file
        piece = load_data(data_path)
        # all_parts is list of melody and chords, each is sequence of [start time, duration, octave, pitch, velocity]
        all_parts=[]
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
            # part_tuples is sequence of [start time, duration, octave, pitch, velocity]
            part_tuples = []
            for event in part._elements:
                # if chords or notes exist recursive (sometimes this happened in a complicated piano song file)
                if event.isStream :
                    _part_tuples = []
                    for i in event._elements:
                        _part_tuples = self.streaming(event, i, _part_tuples)
                    all_parts.append(_part_tuples)
                # normal case
                else:
                    part_tuples = self.streaming(part, event, part_tuples)
            if part_tuples != []:
                all_parts.append(part_tuples)
        # print(piece.analyze('key'))
        parsed = self.compare_parts(all_parts)
        sequence = self.sequentialize(parsed)
        return sequence

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
            # save index for sorting pitches of chord
            sort_idx = [i[0] for i in sorted(enumerate(event.pitchClasses), key=lambda x: x[1])]
            octaves = [x for (y, x) in sorted(zip(sort_idx, octaves))]
            ch_idx = self.chord_ref.index(event.orderedPitchClasses)
            oc_idx = self.octave_ref.index(octaves)
            part_tuples.append([offset, event.quarterLength, oc_idx, ch_idx, event.volume.velocity])

            """
            ch_idx = self.chord_ref.index(event.orderedPitchClasses)
            oc_idx = self.octave_ref.index(octaves)
            self.chords_cnt[ch_idx] += 1
            self.chord_octaves_cnt[oc_idx] += 1
            if event.orderedPitchClasses not in self.chords:
                self.chords.append(event.orderedPitchClasses)
            if octaves not in self.chord_octaves:
                self.chord_octaves.append(octaves)
            """

        # if current event is note
        if getattr(event, 'isNote', None) and event.isNote:
            # change to key
            # make one step in sequence
            """
            if event.pitch.octave not in self.note_octaves:
                self.note_octaves.append(event.pitch.octave)
            if event.pitchClass not in self.notes:
                self.notes.append(event.pitchClass)
            self.notes_cnt[no_idx] += 1
            self.note_octaves_cnt[oc_idx] += 1
            """
            no_idx = self.note_ref.index(event.pitchClass)
            oc_idx = self.note_octave_ref.index(event.pitch.octave)
            part_tuples.append(
                [offset, event.quarterLength, oc_idx, no_idx, event.volume.velocity])
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

    def chk_same(self,melody, chord):
        mel_time = [item[0] for item in melody]
        cho_time = [item[0] for item in chord]
        if mel_time == cho_time:
            return True
        else:
            return False

    def sequentialize(self, parsed):
        if len(parsed[0])!=len(parsed[1]):
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
        seq = a.parsing(data_dir+file)
        dataset.append(seq)
    
    with open('dataset', 'wb') as fp:
        pickle.dump(dataset, fp)

    # fraction to float & for python2
    with open('./dataset/dataset', 'rb') as fp:
        data_fr = pickle.load(fp)
        data_str = []
        for i in data_fr:
            song = []
            for j in i:
                pattern = "%.4f"
                song.append([pattern % k for k in j])
            data_str.append(song)

        data2 = []
        for i in data_str:
            song2 = []
            for j in i:
                song2.append(list(map(float, j)))
            data2.append(song2)

        with open('./dataset/dataset2', 'wb') as fp:
            pickle.dump(data2, fp, protocol=2)
        print('done!')
    


    """ # check dataset
    print('notes: ', a.notes)
    print('note_octaves: ', a.note_octaves)
    print('notes_cnt: ', a.notes_cnt)
    print('note_octaves_cnt: ', a.note_octaves_cnt)
    
    print('chords_cnt: ', a.chords_cnt)
    print('octaves_cnt: ', a.chord_octaves_cnt)
    print('chords: ',a.chords)
    print('octaves: ',a.chord_octaves)
    
    print('len(chords): ',len(a.chords))
    print('len(chord_octaves): ',len(a.chord_octaves))
    print('\n')
    """

