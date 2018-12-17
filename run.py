import preproc
import resources
import parent
import interact_debug
import interact_midi

import sys, os
import platform

from glob import glob



def main():

    set_dir()
    clear_sc()

    while True:

        display_options()
        inp = input('\n> input: ')
        clear_sc()

        if inp == '1' or inp.startswith('s'):
            datasize = resources.get_datasize(parent.data_path)
            print(f'> {datasize}')
            
        elif inp == '2' or inp.startswith('p'):
            new_datasize = preproc.bootstrap()
            print(f'> {new_datasize}')

        elif inp == '3' or inp.startswith('t'):

            # inp essential args
            
            arr = []
            print('> datasize batchsize epochs :')
            while (len(arr) < 3):
                arr.extend(input().split(" "))
            ds, bs, ep = arr[:3]

            # inp optional args

            params, args = [], []

            print('> optional args: ')
            while True:
                inp = input('(hit enter when done) ')
                if inp == '':
                    break
                else:
                    try:

                        stuff = inp.split(" ")
                        for s in stuff:
                            arr = s.split("=")

                            if len(arr) == 2:
                                params.append(arr[0])
                                args.append(arr[1])

                    except: print("correct format: param1=arg1 param2=arg2 ..")

            # connections to parent

            for p,a in zip(params,args):
                if p == 'lr1': parent.learning_rate_1 = float(a)
                elif p == 'lr2': parent.learning_rate_2 = float(a)
                elif p == 'startadv': parent.start_advanced = bool(a)
                elif p == 'adv': parent.further_parenting = bool(a)
                elif p == 'drop': parent.dropout = float(a)
                elif p == 'onlyloss':parent.only_loss_on = tuple(int(a)-1 for a in a.split(','))
            clear_sc();

            parent.bootstrap(True, int(ep), int(ds), int(bs))
                    
        
        elif inp == '4' or inp.startswith('m'):
            interact_midi.bootstrap()

        elif inp == '5' or inp.startswith('i'):
            # interact.bootstrap() # will come someday
            pass

        elif inp.startswith('g'):
            resources.graph_bootstrap()

        elif inp == 'manual':
            interact_debug.bootstrap()

        elif inp.startswith('d'):
            
            display_debug_options()
            inp = input('\n> debug-input: ')
            clear_sc()

            if inp == '1':
                try:
                    restore_from_latest_ckpt()
                except Exception as e: print('Op error: ', e)

            if inp == '2':
                try:
                    revert_to_before_session()
                except Exception as e: print('Op error: ', e)

            elif inp == '3' and input('Removes intermediate files, continue? (y/n): ').lower() == 'y':
                try:
                    remove_intermediate_data()
                except Exception as e: print('Op error: ', e)

            
            elif inp == '4' and input('Removes processed work, continue? (y/n): ').lower() == 'y':
                try:
                    remove_preprocessed_data()
                except Exception as e: print('Op error: ', e)


            if inp == '5' and input('Cleans up model data continue? (y/n): ').lower() == 'y':
                try:
                    remove_training_data()
                except Exception as e: print('Op error: ', e)


        elif inp == '0': break
        else: pass

        input('\n Hit Enter to continue..')
        clear_sc()


def display_options():

    print('\n \t Options: \n')
    print('(1)- Current datasize')
    print('(2)- Preprocess samples')
    print('(3)- Train model')
    print('(4)- Midi response')
    print('(5)- Interact')
    print('0 - exit.')
    print('---------')

def display_debug_options():

    print('\n \t\t Debug Menu: \n')
    print('1 - Update from last checkpoint.')
    print('2 - Revert to previous session.')
    print('3 - Clear checkpoints.')
    print('4 - Clear .pkls')
    print('5 - Remove model. ')
    print('0- return.')
    print('---------')

def set_dir():
    currdir = os.path.dirname(sys.argv[0])
    os.chdir(os.path.abspath(currdir))

def clear_sc():
    os.system("cls" if platform.system().lower() == "windows" else "clear")



main_files = ['model.pkl',
              'model_accugrads.pkl',
              'model_moments.pkl']

intermediate_files = ['model0*.pkl',
                      'model_accugrads0*.pkl',
                      'model_moments0*.pkl']

all_model_files = ['model*.pkl']

session_model_files = ['model_before_simple.pkl',
                       'model_before_advanced.pkl']


def restore_from_latest_ckpt():
    ckpts = []
    
    for file in intermediate_files:
        try: ckpts.append(max(glob(file), key=os.path.getmtime))
        except: ckpts.append(None)

    for name, item in zip(main_files, ckpts):
        if item is not None and input(f'Restore {item} -> {name}? y/n: ').lower() == 'y':
            try: os.remove(name)
            except: pass
            os.rename(item, name)
            print('restored.')

def revert_to_before_session():
    sess_files = []

    for item in session_model_files:
        sess_files.extend(glob(item))

    if sess_files != []:

        last_sess_item = max(sess_files, key=os.path.getmtime)
        if input(f'Restore {last_sess_item} -> model.pkl? y/n: ').lower() == 'y':

            try: os.remove('model.pkl')
            except: pass
            finally:
                os.rename(last_sess_item, 'model.pkl')
                print('restored.')

    else: print("No previous session data found.")

def remove_training_data():
    import os.path
    to_remove = []
    to_remove.extend(glob('meta*.pkl'))
    to_remove.extend(glob('model*.pkl'))
    to_remove.extend(glob('loss*.txt'))
    for e in to_remove:
        if os.path.exists(e):
            os.remove(e)
            print(f'removed {e}.')

def remove_preprocessed_data():
    to_remove = glob('sample*.pkl')
    for e in to_remove:
        os.remove(e)
        print(f'removed {e}.')

def remove_intermediate_data():
    for something in intermediate_files:
        things = glob(something)
        for it in things:
            os.remove(it) ; print(f'Removed {it}.')
    for leftover in all_model_files:
        leftovers = glob(leftover)
        [os.remove(this) for this in leftovers if this not in main_files]





if __name__ == '__main__':main()
