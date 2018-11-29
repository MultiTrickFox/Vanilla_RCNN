import preproc
import resources
import parent
import interact_debug
import interact_midi

import sys, os
import platform




def main():

    currdir = os.path.dirname(sys.argv[0])
    os.chdir(os.path.abspath(currdir))
    clear_sc()

    while True:
        display_options()
        inp = input('> input: ')
        clear_sc()

        if inp == '1':
            print(f'> {resources.get_datasize(parent.data_path)}')
        elif inp == '2':
            preproc.bootstrap()

        elif inp == '3':
            
            arr = []
            print('<datasize> <batchsize> <epochs>: ', end="")
            while (len(arr) < 3):
                arr.extend(input().split(" "))
            ds, bs, ep = arr[:3]

            params, args = [], []
            while(True):
                inp = input('optional args: (hit enter when done): ')
                if inp == '':
                    break
                else:
                    try:
                        arr = inp.split("=")
                        if (len(arr) == 2): 
                            params.append(arr[0])
                            args.append(arr[1])
                    except Exception as e: print('weird input.', e)

            for p,a in zip(params,args):
                if p == 'lr1': parent.learning_rate_1 = float(a)
                elif p == 'lr2': parent.learning_rate_2 = float(a)
                elif p == 'startadv': parent.start_advanced = bool(a)
                elif p == 'adv': parent.further_parenting = bool(a)
                elif p == 'drop': parent.dropout = float(a)
                else: print('Available params: lr1,lr2,startadv,adv,drop')

            parent.bootstrap(True, int(ep), int(ds), int(bs))
        
        elif inp == '4':
            interact_midi.bootstrap()

        elif inp == '5':
            # interact.bootstrap() # will come someday
            pass

        elif inp == 'manual':
            interact_debug.bootstrap()

        elif inp == 'debug':
            
            display_debug_options()

            while(True):

                inp = input('>debug-input: ')
                clear_sc()

                if inp == '1' and input('Removes trained model, continue? (y/n): ').lower() == 'y':
                    try: remove_training_data()
                    except Exception as e: print('Remove error: ', e)

                
                elif inp == '2' and input('Removes .pkl data, continue? (y/n): ').lower() == 'y':
                    try: remove_preprocess_data()
                    except Exception as e: print('Remove error: ', e)

                else: break

        elif inp == '0': break
        else: pass

        input('Hit any key to continue..')
        clear_sc()


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
