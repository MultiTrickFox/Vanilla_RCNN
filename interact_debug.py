import Vanilla
import resources
import preproc

import torch


max_octave = preproc.MAX_OCTAVE
max_duration = preproc.MAX_DURATION
max_volume = preproc.MAX_VOLUME

pick_thr = (1-0.3)

write_response = True



def bootstrap(input_sequence=None, noprint=False, multipropogate=0):
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

        response = Vanilla.forward_prop(model, input_sequence)

        converted_response = [conv(out_t) for out_t in response]

        with open('responses.txt','a+')as file:

            for resp_conv, resp_act in zip(converted_response, response):

                if not noprint:
                    print('---')
                    print(' Notes:', resp_conv[0], resp_act[0], end=" ")
                    print(' Octaves:', resp_conv[1], resp_act[1], end=" ")
                    print(' Durations:', resp_conv[2], resp_act[2], end=" ")
                    print(' Velocities:', resp_conv[3], resp_act[3], end=" ")
                    print('---')
                
                for e in resp_act:
                    file.write(str(e.detach().numpy())+' \n')
                file.write('\n')

        print(f'Response length: {len(converted_response)}')

    else:

        response = []
        # input_sequence = [[torch.Tensor(e) for e in sequence_t] for sequence_t in input_sequence]
        sequence = resources.tensorify_sequence(input_sequence)
        output = Vanilla.forward_prop(model, sequence)
        response.extend(output)
        for _ in range(multipropogate):
            output = Vanilla.forward_prop(model, output)
            response.extend(output)

        converted_response = [conv(out_t) for out_t in response]

        if write_response: [write_response_txt(t) for t in response]

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
    
    if sel_vocabs == []:

        first, second, third = 0,0,0
        first_i, second_i, third_i = 0,0,0

        for _,v in enumerate(vocabs):

            if v > first:
                third = second ; third_i = second_i
                second = first ; second_i = first_i
                first = v      ; first_i = _

            elif v > second:
                third = second ; third_i = second_i
                second = v     ; second_i = _

            elif v > third:
                third = v      ; third_i = _
        
        sel_vocabs = [first_i, second_i, third_i]

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


def write_response_txt(response_t):
    with open('responses.txt',"a+") as file:
        for result_element in response_t:
            result_element = "".join([str(float(e))+"," for e in result_element])
            file.write(result_element+";")
        file.write('\n\n')
            # file.write(e.detach().numpy())



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

