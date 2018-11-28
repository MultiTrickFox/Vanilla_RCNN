import interact_debug
import preproc

import glob


def bootstrap():


    try:
        file = glob.glob("*.mid")[-1]
        print(f'Obtained file: {file}')
    except Exception as e:
        print('Error : No .mid files found.')
        file = input('Drag & Drop your midi here / enter path: ')

    try:
        if file is not None:
            print('Processing data..')
            data = preproc.midi_to_stream(file)
        else: data = None
    except Exception as e:
        data = None
        print('Error : Provided file could not be processed.',e)

    try:
        if data is not None:
            print('Asking ai..')
            response = interact_debug.bootstrap(data)

            print("-----")
            for resp_step in response:
                for stuff in resp_step:
                    print(stuff)
                print("-----")
    except Exception as e: print('Error : No response could be generated for given file.',e)


if __name__ == '__main__':
    bootstrap()
    input('Hit any key to continue..')
