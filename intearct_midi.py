import interact_debug
import preproc

import glob

pick_thr = \
float(input("Decision Threshold: "))

print('Processing data..')
data = preproc.parse_to_data(glob.glob("*.mid")[-1])
print('Asking ai..')
response = interact_debug.bootstrap(data, pick_thr)

print("-----")
for resp_step in response:
    for stuff in resp_step:
        print(stuff)
    print("-----")
input()
