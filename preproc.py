import resources

import gc

from multiprocessing import Pool, cpu_count
from glob import glob

from music21 import converter

#   params

SPLIT_DURATION = 2

min_phrase_len = 5
max_phrase_len = 25

MAX_OCTAVE = 7
MAX_DURATION = 8
MAX_VOLUME = 127

hm_batches = 15

show_passed_exceptions = False

#   global dec

cpu_count = cpu_count()

note_dict, note_reverse_dict = \
    resources.note_dict, resources.note_reverse_dict

vocab_size = len(note_reverse_dict)
empty_vect = [0 for _ in range(vocab_size)]



def preproc(raw_file):
    imported_file = import_file(raw_file)
    if imported_file is not None:
        converted_file = convert_file(imported_file)
        if len(converted_file) != 0:
            data = parse_file(converted_file)
            if len(data[0]) != 0:
                return data
    return [[],[]]

    # return parse_file(convert_file(import_file(raw_file)))


def preprocess(raw_files):
    data = [[],[]]

    with Pool(cpu_count) as p:

        results = p.map_async(preproc, raw_files)

        p.close()
        p.join()

        for result in results.get():
            for e,re in zip(data,result):
                if len(re)>= 0: e.append(re)

    return data


def import_file(raw_file):
    try:
        prepared_file = converter.parse(raw_file)

        # def_mtr = music21.meter.TimeSignature('4/4')
        # for element in stream:
        #     if type(element) is music21.meter.TimeSignature:
        #         del element
        # stream.insert(0, def_mtr)
        # @todo: find a way to auto-conv everything to 4/4

        imported_file = prepared_file.flat.elements
        return imported_file
    except Exception as e:
        if show_passed_exceptions: print(f'Error : File {raw_file} : {e}')
        return None


def convert_file(imported_file):
    converted_file = []

    for element in imported_file:
        element_vector = vectorize_element(element)

        if element_vector[0] is not None:
            converted_file.append(element_vector)

    return converted_file


def parse_file(stream):

    phrase_container = []
    phrase_stream = []

    for e in stream:
        phrase_container.append(e)

        if split_cond(e[2]):

            if min_phrase_len <= len(phrase_container) <= max_phrase_len:
                phrase_stream.append(phrase_container)

    data = [[],[]]  # data = phrases_X & phrases_Y

    if len(phrase_stream)>= 2:
        for i, thing in enumerate(phrase_stream[:-1]):
            thingp1 = phrase_stream[i+1]
            # for container, phrase in zip(data, [thing, thingp1]):
            #     container.append(phrase)
            data[0].append(thing)
            data[1].append(thingp1)

    return data

    # each phrase_X is a sequence, w 4 elements per step)


# helpers


def split_cond(dur_vect):
    for dur in dur_vect:
        if dur >= SPLIT_DURATION / MAX_DURATION: return True
    return False


def hasValid_duration(element): return 0.0 < float(element.duration.quarterLength) <= MAX_DURATION


def vectorize_element(element):

    vocab_vect = empty_vect.copy()
    oct_vect   = vocab_vect.copy()
    dur_vect   = vocab_vect.copy()
    vol_vect   = vocab_vect.copy()

    try:
        if element.isNote:
            note_id = note_dict[element.pitch.name]
            if hasValid_duration(element):
                vocab_vect[note_id] += 1
                oct_vect[note_id] += float(element.pitch.octave)
                dur_vect[note_id] += float(element.duration.quarterLength)
                vol_vect[note_id] += float(element.volume.velocity)

        elif element.isChord:
            for e in element:
                note_id = note_dict[e.pitch.name]
                if hasValid_duration(e):
                    duplicateNote = vocab_vect[note_id] != 0
                    vocab_vect[note_id] += 1
                    oct_vect[note_id] += float(e.pitch.octave)
                    dur_vect[note_id] += float(e.duration.quarterLength)
                    vol_vect[note_id] += float(e.volume.velocity)

                    if duplicateNote:
                        oct_vect[note_id] /= 2
                        dur_vect[note_id] /= 2
                        vol_vect[note_id] /= 2

        elif element.isRest:
            if hasValid_duration(element):
                note_id = note_dict['R']
                vocab_vect[note_id] += 1
                dur_vect[note_id] += float(element.duration.quarterLength)

        # normalization & final-fixes

        vocab_sum = sum(vocab_vect)

        if vocab_sum == 0: return None, None, None, None

        if vocab_sum != 1: vocab_vect = [round(float(e / vocab_sum), 3) for e in vocab_vect]
        oct_vect = [round(e / MAX_OCTAVE, 3) for e in oct_vect]
        dur_vect = [round(e / MAX_DURATION, 3) for e in dur_vect]
        vol_vect = [round(e / MAX_VOLUME, 3) for e in vol_vect]

    except Exception as e:
        if show_passed_exceptions: print(f'Error : Element {element} : {e}')
        return None, None, None, None

    return vocab_vect, oct_vect, dur_vect, vol_vect






def bootstrap():

    raw_files = glob('samples/*.mid')
    hm_files = len(raw_files)
    data_size = 0

    print(f'Detected {hm_files} files.')
    batch_len = int(hm_files / hm_batches)

    for batch_id, files in enumerate(resources.batchify(raw_files, batch_len)):

        data = preprocess(raw_files=files)

        resources.pickle_save(data, 'samples_' + str(batch_id) + '.pkl')
        data_size += len(data)
        data = None ; gc.collect()

        print(f'Batch {batch_id} of {hm_batches} completed : Total of {data_size} samples.')
    print(f'Total of {data_size} samples saved.')
    return data_size





if __name__ == '__main__':
    bootstrap()
