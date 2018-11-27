import interact_debug
import preproc

import glob


def bootstrap():


    try:
        file = glob.glob("*.mid")[-1]
        print(f'Obtained file: {file}')
    except:
        print('Error : No .mid files found.')
        file = None

    try:
        if file is not None:
            print('Processing data..')
            data = preproc.midi_to_stream(file)
        else: data = None
    except:
        data = None
        print('Error : Provided file could not be processed.')

    if data is not None:
        print('Asking ai..')
        response = interact_debug.bootstrap(data)

        print("-----")
        for resp_step in response:
            for stuff in resp_step:
                print(stuff)
            print("-----")


if __name__ == '__main__':
    bootstrap()
    input('Hit any key to continue..')
