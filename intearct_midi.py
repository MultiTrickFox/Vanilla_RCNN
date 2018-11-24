import interact_debug
import preproc

interact_debug.vocab_pick_thr = 0.3


import glob

data = preproc.parse_to_data(glob.glob("respond_to*.mid")[-1])
response = interact_debug.bootstrap(data)

for resp_step in response:
    for stuff in resp_step:
        print(stuff)
    print("-----")
