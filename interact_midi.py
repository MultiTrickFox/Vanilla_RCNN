import interact_debug
import preproc

import glob



def bootstrap():


    try:

        file = glob.glob("*.mid")[-1]
        print(f'Obtained file: {file}')

    except Exception as e:

        print('Error : No .mid files found.')
        file = None
        # file = os.path.abspath(input('Drag & Drop midi file: '))

    try:

        if file is not None:
            print('Processing data..')
            data = preproc.midi_to_stream(file)

        else: data = []

    except Exception as e:

        data = []
        print('Error : Provided midi cannot be processed.',e)

    try:

        if len(data) > 0:

            print('Asking ai..')
            response = interact_debug.bootstrap(data)

            print("-----")
            for resp_step in response:
                for stuff in resp_step:
                    print(stuff)
                print("-----")

        else: print('Error : No midi information was extracted.')

    except Exception as e: print('Error : No response could be generated.',e)





if __name__ == '__main__':
    bootstrap()
    input('Hit any key to continue..')
