import res
import glob
from music21 import converter
from multiprocessing import Pool, cpu_count


min_seq_len = 5
max_seq_len = 30

MAX_OCTAVE = 7
MAX_DURATION = 8
SPLIT_DURATION = 2  # of 16th notes
MAX_VOLUME = 127

show_passed_exceptions = False


def preprocess(raw_files=glob.glob("samples/*.mid")):

    imported_files = []

    print(f'\nDetected CPU(s): {cpu_count()}\n')

    with Pool(cpu_count()) as p:

        results = p.map_async(import_fn, raw_files)

        p.close()
        p.join()

        for result in results.get():
            if result is not None:
                imported_files.append(result)
        print(f'Files Obtained: {len(results.get())}\n')

    vocab_seqs_X, vocab_seqs_Y = [], []
    oct_seqs_X, oct_seqs_Y = [], []
    dur_seqs_X, dur_seqs_Y = [], []
    vol_seqs_X, vol_seqs_Y = [], []

    with Pool(cpu_count()) as p2:

        results = p2.map_async(parse_fn, imported_files)

        p2.close()
        p2.join()

        for result in results.get():
            if len(result[0]) is not 0:
                vocab_seqs_X.extend(result[0])
                vocab_seqs_Y.extend(result[1])
                oct_seqs_X.extend(result[2])
                oct_seqs_Y.extend(result[3])
                dur_seqs_X.extend(result[4])
                dur_seqs_Y.extend(result[5])
                vol_seqs_X.extend(result[6])
                vol_seqs_Y.extend(result[7])
        print()

    len_samples = len(vocab_seqs_X)

    data = [
        [vocab_seqs_X, oct_seqs_X, dur_seqs_X, vol_seqs_X],
        [vocab_seqs_Y, oct_seqs_Y, dur_seqs_Y, vol_seqs_Y]]

    print(f'Samples Collected: {len_samples}\n')

    return data, len_samples


def parse_fn(stream):
    vocab_seqs_X, vocab_seqs_Y = [], []
    oct_seqs_X, oct_seqs_Y = [], []
    dur_seqs_X, dur_seqs_Y = [], []
    vol_seqs_X, vol_seqs_Y = [], []

    mstream = []

    vocab_seq_container = []
    oct_seq_container = []
    dur_seq_container = []
    vol_seq_container = []

    for element in stream:

        vocab_vect, oct_vect, dur_vect, vol_vect = vectorize_element(element)

        if vocab_vect is not None:

            vocab_seq_container.append(vocab_vect)
            oct_seq_container.append(oct_vect)
            dur_seq_container.append(dur_vect)
            vol_seq_container.append(vol_vect)

            if split_cond(dur_vect):
                
                if min_seq_len <= len(vocab_seq_container) <= max_seq_len:
                    mstream.append([vocab_seq_container,
                                    oct_seq_container,
                                    dur_seq_container,
                                    vol_seq_container])

                vocab_seq_container, oct_seq_container, dur_seq_container, vol_seq_container = [], [], [], []

    if len(mstream) != 1:
        for i, thing in enumerate(mstream[:-1]):
            thingp1 = mstream[i + 1]
            vocab_seqs_X.append(thing[0])
            oct_seqs_X.append(thing[1])
            dur_seqs_X.append(thing[2])
            vol_seqs_X.append(thing[3])
            vocab_seqs_Y.append(thingp1[0])
            oct_seqs_Y.append(thingp1[1])
            dur_seqs_Y.append(thingp1[2])
            vol_seqs_Y.append(thingp1[3])

    # print('File Parsed.')
    return \
        vocab_seqs_X, vocab_seqs_Y, \
        oct_seqs_X, oct_seqs_Y, \
        dur_seqs_X, dur_seqs_Y, \
        vol_seqs_X, vol_seqs_Y


def vectorize_element(element):
    vocab_vect = [0 for _ in range(vocab_size)]
    oct_vect = vocab_vect.copy()
    dur_vect = vocab_vect.copy()
    vol_vect = vocab_vect.copy()

    try:
        if element.isNote:
            note_id = note_dict[element.pitch.name]
            if duration_isValid(element):
                vocab_vect[note_id] += 1
                oct_vect[note_id] += float(element.pitch.octave)
                dur_vect[note_id] += float(element.duration.quarterLength)
                vol_vect[note_id] += float(element.volume.velocity)

        elif element.isChord:
            for e in element:
                note_id = note_dict[e.pitch.name]
                if duration_isValid(e):
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
            if duration_isValid(element):
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
        if show_passed_exceptions: print('Element', element, 'passed Error:', e)
        return None, None, None, None

    return vocab_vect, oct_vect, dur_vect, vol_vect


def duration_isValid(element): return 0.0 < float(element.duration.quarterLength) <= MAX_DURATION


note_dict = res.note_dict

note_reverse_dict = res.note_reverse_dict

vocab_size = len(note_reverse_dict)


def split_cond(dur_vect):
    for dur in dur_vect:
        if dur >= SPLIT_DURATION / MAX_DURATION: return True
    return False


def import_fn(raw_file):
    try:
        raw_stream = converter.parse(raw_file)
        stream = ready_stream(raw_stream)
        return stream
    except:
        return None
    # finally: print('file scanned.')


def ready_stream(stream):
    # def_mtr = music21.meter.TimeSignature('4/4')
    # for element in stream:
    #     if type(element) is music21.meter.TimeSignature:
    #         del element
    # stream.insert(0, def_mtr)
    # todo: find a way to auto-conv everything to 4/4

    return stream.flat.elements


empty_vect = [0 for _ in range(vocab_size)]





def parent_bootstrap():
    raw_files = glob.glob("samples/*.mid")
    hm_files = len(raw_files)
    print(f'Sample Files: {hm_files} detected.')
    batch_len = int(hm_files / 10)
    hm_batches = int(hm_files / batch_len)
    for _ in range(hm_batches):
        ptr1 = int(_ * batch_len)
        ptr2 = int((_ + 1) * batch_len)
        files = raw_files[ptr1:ptr2]
        data, __ = preprocess(raw_files=files)
        res.pickle_save(data, 'samples_' + str(_) + '.pkl')
        data = []
        print(f'Batch {_} of {hm_batches} completed.')



if __name__ == '__main__':
    parent_bootstrap()
