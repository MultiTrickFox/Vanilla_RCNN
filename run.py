import preproc
import resources
import parent
import interact_midi



def main():


    while True:
        display_options()
        inp = input('>input: ')

        if inp == '1':
            print(resources.get_datasize())
        elif inp == '2':
            preproc.bootstrap()
        elif inp == '3':
            str = input('<epoch>,<datasize>,<batchsize> ')
            ep, ds, bs = [int(e) for e in str.split(",")]
            parent.bootstrap(True, ep, ds, bs)
        elif inp == '4':
            interact_midi.bootstrap()
        else:
            pass # will come someday


def display_options():
    print('1- Get current datasize')
    print('2- Preprocess samples')
    print('3- Train <Epochs, Datasize, Batchsize>')
    print('4- Process .mid file')
    print('5- Interact')
    print()


if __name__ == '__main__':main()
