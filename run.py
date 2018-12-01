import preproc
import resources
import parent
import interact_debug
import interact_midi

import sys, os
import platform

from glob import glob



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

            # inp essential args
            
            arr = []
            print('> datasize batchsize epochs :')
            while (len(arr) < 3):
                arr.extend(input().split(" "))
            ds, bs, ep = arr[:3]

            # inp optional args

            params, args = [], []
            print('> optional args: (hit enter when done) ')
            while(True):
                inp = input()
                if inp == '':
                    break
                else:
                    try:
                        stuff = inp.split(" ")
                        for s in stuff:
                            arr = s.split("=")
                            if (len(arr) == 2): 
                                params.append(arr[0])
                                args.append(arr[1])
                            else: print("correct format: param1=arg1 param2=arg2 ..")
                    except Exception as e: print("correct format: param1=arg1 param2=arg2 ..")

            # connections to parent

            for p,a in zip(params,args):
                if p == 'lr1': parent.learning_rate_1 = float(a)
                elif p == 'lr2': parent.learning_rate_2 = float(a)
                elif p == 'startadv': parent.start_advanced = bool(a)
                elif p == 'adv': parent.further_parenting = bool(a)
                elif p == 'drop': parent.dropout = float(a)
                elif p == 'onlyloss': parent.only_loss_on = int(a)
                else: print('Available params: lr1,lr2,startadv,adv,drop,onlyloss')

            clear_sc();

            parent.bootstrap(True, int(ep), int(ds), int(bs))
                    
        
        elif inp == '4':
            interact_midi.bootstrap()

        elif inp == '5':
            # interact.bootstrap() # will come someday
            pass

        elif inp == 'graph':
            resources.graph_bootstrap()

        elif inp == 'manual':
            interact_debug.bootstrap()

        elif inp == 'debug':
            
            display_debug_options()

            while(True):

                inp = input('>debug-input: ')
                clear_sc()

                if inp == '1':
                    try:
                        restore_from_checkpoint()
                        print('done.')
                    except Exception as e: print('Op error: ', e)

                elif inp == '2' and input('Removes intermediate data, continue? (y/n): ').lower() == 'y':
                    try:
                        remove_intermediate_data()
                        print('done.')
                    except Exception as e: print('Op error: ', e)

                
                elif inp == '3' and input('Removes .pkl data, continue? (y/n): ').lower() == 'y':
                    try:
                        remove_preprocess_data()
                        print('done.')
                    except Exception as e: print('Op error: ', e)


                if inp == '4' and input('Removes trained model, continue? (y/n): ').lower() == 'y':
                    try:
                        remove_training_data()
                        print('done.')
                    except Exception as e: print('Op error: ', e)


                else: break

        elif inp == '0': break
        else: pass

        input('\n Hit any key to continue..')
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
    print('1- Restore from checkpoint.')
    print('2- Remove checkpoint data.')
    print('3- Remove data .pkls')
    print('4- Remove model. ')
    
    print('0- Return')
    # print('4- Midi response')
    # print('5- Interact')
    # print('0- ')
    print('---------')

def clear_sc():
    os.system("cls" if platform.system().lower() == "windows" else "clear")



main_files = ['model.pkl',
              'model_accugrads.pkl',
              'model_moments.pkl',
              'meta.pkl']

intermediate_files = ['model0*.pkl',
                      'model_accugrads0*.pkl',
                      'model_moments0*.pkl',
                      'meta0*.pkl']


def restore_from_checkpoint():

    try: model_ckpt = max(glob('model0*.pkl'), key=os.path.getctime)
    except: model_ckpt = None
    try: accugrad_ckpt = max(glob('model_accugrads0*.pkl'), key=os.path.getctime)
    except: accugrad_ckpt = None
    try: moment_ckpt = max(glob('model_moments0*.pkl'), key=os.path.getctime)
    except: moment_ckpt = None
    try: meta_ckpt = max(glob('meta0*.pkl'), key=os.path.getctime)
    except: meta_ckpt = None
    for name, item in zip(main_files, [model_ckpt, accugrad_ckpt, moment_ckpt, meta_ckpt]):
        if item is not None and input(f'Restore {item} -> {name}? y/n: ').lower() == 'y':
            os.remove(name)
            os.rename(item, name)
            print(f'restored.')

def remove_training_data():
    import os.path
    to_remove = []
    to_remove.extend(glob('meta*.pkl'))
    to_remove.extend(glob('model*.pkl'))
    to_remove.extend(glob('*.txt'))
    for e in to_remove:
        if os.path.exists(e):
            os.remove(e)
            print(f'removed {e}.')

def remove_preprocess_data():
    to_remove = glob('sample*.pkl')
    for e in to_remove:
        os.remove(e)
        print(f'removed {e}.')

def remove_intermediate_data():
    for something in intermediate_files:
        things = glob(something)
        for it in things:
            os.remove(it) ; print(f'Removed {it}.')    





if __name__ == '__main__':main()
