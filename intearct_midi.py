import interact_debug
import preproc

pick_thr = 0.3


import glob

data = preproc.midi_to_data(glob.glob("*.mid")[-1])
response = interact_debug.bootstrap(data, pick_thr)

for resp_step in response:
    for stuff in resp_step:
        print(stuff)
    print("-----")
