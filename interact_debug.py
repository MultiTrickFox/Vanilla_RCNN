import Vanilla
import resources
import preproc

import torch


max_octave = preproc.MAX_OCTAVE
max_duration = preproc.MAX_DURATION
max_volume = preproc.MAX_VOLUME

pick_thr = (1-0.3)



def bootstrap(input_sequence=None):
    global pick_thr

    model = resources.load_model()
    while model is None:
        model_id = input('Import model and Hit Enter.. ')
        model = resources.load_model() if model_id == '' else resources.load_model(model_id)


    chord_mode = input("hit 's' for Chord Mode -> Solo Mode: ")
    chord_mode = False if chord_mode == 's' else True
    if chord_mode:
        pick_thr_inp = input("Chord Decision Sensitivity: ")
        if pick_thr_inp != "": pick_thr = 1-float(pick_thr_inp)


    conv = lambda output: ai_2_human(output, chord_mode=chord_mode, pick_thr=pick_thr)


    if input_sequence is None:

        input_sequence = get_user_input(int(input('Enter an Input Length: ')))

        responses = Vanilla.forward_prop(model, input_sequence)

        converted_response = [conv(out_t) for out_t in responses]

        for response in converted_response:

            print('---')
            print(' Notes:', response[0])
            print(' Octaves:', response[1])
            print(' Durations:', response[2])
            print(' Velocities:', response[3])
            print('---')

        print(f'Response length: {len(converted_response)}')

    else:

        # input_sequence = [[torch.Tensor(e) for e in sequence_t] for sequence_t in input_sequence]
        sequence = resources.tensorify_sequence(input_sequence)
        response = Vanilla.forward_prop(model, sequence)
        converted_response = [conv(out_t) for out_t in response]

    return converted_response
    



# helper-converters


def ai_2_human(out_t, chord_mode=True, pick_thr=pick_thr):

    vocabs, octaves, durations, volumes = out_t

    # sel_vocabs = []
    sel_octs   = []
    sel_durs   = []
    sel_vols   = []

    sel_vocabs = [_ for _,e in enumerate(vocabs) if e.item() >= pick_thr] \
        if chord_mode else [torch.argmax(vocabs).item()]

    for vocab in sel_vocabs:
        sel_octs.append(round(float(octaves[vocab]) * max_octave))
        sel_durs.append(round(float(durations[vocab]) * max_duration, 2))
        sel_vols.append(round(float(volumes[vocab]) * max_volume))

    sel_vocabs = [resources.note_reverse_dict[_] for _ in sel_vocabs]

    return sel_vocabs, sel_octs, sel_durs, sel_vols


def human_2_ai(data):

    notes, octaves, durations, volumes = data
    inp_len = len(notes)

    c_notes = [resources.note_dict[notes[i].upper()] for i in range(inp_len)]

    vocab_vect = preproc.empty_vect.copy()
    oct_vect   = preproc.empty_vect.copy()
    dur_vect   = preproc.empty_vect.copy()
    vol_vect   = preproc.empty_vect.copy()

    for i, note in enumerate(c_notes):
        duplicate_note = (vocab_vect[note] != 0)
        vocab_vect[note] += 1
        oct_vect[note]   += float(octaves[i])
        dur_vect[note]   += float(durations[i])
        vol_vect[note]   += float(volumes[i])
        if duplicate_note:
            oct_vect[note]   /= 2
            dur_vect[note]   /= 2
            vol_vect[note]   /= 2

    # normalization & fixes

    vocab_sum = sum(vocab_vect)

    if vocab_sum != 1: vocab_vect = [e/vocab_sum for e in vocab_vect]
    oct_vect = [e/max_octave for e in oct_vect]
    dur_vect = [e/max_duration for e in dur_vect]
    vol_vect = [e/max_volume for e in vol_vect]

    return vocab_vect, oct_vect, dur_vect, vol_vect


# other helpers


def get_user_input(inp_len):
    vocab_seq = []
    oct_seq   = []
    dur_seq   = []
    vol_seq   = []

    for i in range(inp_len):
        notes = str(input('Enter a tone / chord : ')).split(' ')
        octs = str(input('Enter octaves : ')).split(' ')
        durs = str(input('Enter durations : ')).split(' ')
        vols = str(input('Enter volumes : ')).split(' ')

        data = [notes, octs, durs, vols]
        vocab_vect, oct_vect, dur_vect, vol_vect = human_2_ai(data)

        vocab_seq.append(vocab_vect)
        oct_seq.append(oct_vect)
        dur_seq.append(dur_vect)
        vol_seq.append(vol_vect)

    sequence = []

    for _ in range(len(vocab_seq)):
        sequence.append([torch.Tensor(e) for e in [vocab_seq[_], oct_seq[_], dur_seq[_], vol_seq[_]]])


    return sequence





if __name__ == '__main__':
    bootstrap()

