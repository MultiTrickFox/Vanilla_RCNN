import interact_debug
import preproc

import glob

pick_thr = \
    input("Decision Threshold: ")
pick_thr = float(pick_thr) if pick_thr != "" \
    else interact_debug.vocab_pick_thr

try:
    file = glob.glob("*.mid")[-1]
    print(f'Obtained file: {file}')
except:
    print('Error : No .mid files found.')
    file = None

try:
    if file is not None:
        print('Processing data..')
        data = preproc.midi_to_data(file)
    else: data = None
except:
    data = None
    print('Error : Provided file could not be processed.')

try:
    if data is not None:
        print('Asking ai..')
        response = interact_debug.bootstrap(data, pick_thr)

        print("-----")
        for resp_step in response:
            for stuff in resp_step:
                print(stuff)
            print("-----")
except: print('Error : No response could be generated for given file.')


input()
