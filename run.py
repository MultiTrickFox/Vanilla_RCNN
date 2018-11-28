import preproc
import resources
import parent
import interact_debug
import interact_midi

import os
import platform



def main():


    while True:
        display_options()
        inp = input('>input: ')
        clear_sc()

        if inp == '1':
            print(f'> {resources.get_datasize(parent.data_path)}')
        elif inp == '2':
            try: import music21
            except: print('Music21 not found, use : pip3 install -Iv music21==5.1.0')
            finally: preproc.bootstrap()
        elif inp == '3':
            str = input('<datasize> <batchsize> <epochs>: ')
            ds, bs, ep = [int(e) for e in str.split(" ")]
            parent.bootstrap(True, ep, ds, bs)
        elif inp == '4':
            try: import music21
            except: print('Music21 not found, use : pip3 install -Iv music21==5.1.0')
            finally: interact_midi.bootstrap()
        elif inp == '5':
            # interact.bootstrap() # will come someday
            pass

        elif inp == 'manual':
            interact_debug.bootstrap()

        elif inp == 'debug':
            # clear_sc()
            display_debug_options()

            while(True):

                inp = input('>debug-input: ')

                if inp == '1' and input('Removes trained model, continue? (y/n): ').lower() == 'y':
                    try: remove_training_data()
                    except Exception as e: print('Remove error: ', e)

                
                elif inp == '2' and input('Removes .pkl data, continue? (y/n): ').lower() == 'y':
                    try: remove_preprocess_data()
                    except Exception as e: print('Remove error: ', e)

                else: break

        elif inp == '0': break
        else: pass


def display_options():

    print('\n \t Options: \n')
    print('1- Current datasize')
    print('2- Preprocess samples')
    print('3- Train model')
    print('4- Midi response')
    print('5- Interact')
    print('0- ')
    print('---------')

def display_debug_options():

    print('\n \t\t Debug Menu: \n')
    print('1- Remove model. ')
    print('2- Remove data pickles.')
    print('0- Return')
    # print('4- Midi response')
    # print('5- Interact')
    # print('0- ')
    print('---------')

def clear_sc():
    print(platform.system())
    os.system("cls" if platform.system().lower() == "windows" else "clear")

def remove_training_data():
    import os.path
    from glob import glob
    to_remove = ['last_loss.pkl']
    to_remove.extend(glob('model*.pkl'))
    to_remove.extend(glob('loss*.txt'))
    for e in to_remove:
        if os.path.exists(e):
            os.remove(e)
            print(f'removed {e}.')

def remove_preprocess_data():
    import os
    from glob import glob
    to_remove = glob('sample*.pkl')
    for e in to_remove:
        os.remove(e)
        print(f'removed {e}.')





if __name__ == '__main__':main()
