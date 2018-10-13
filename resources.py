import glob
import pickle
import random
import music21

from music21 import converter # note, chord
from multiprocessing import Pool, cpu_count



note_dict = {
    'A' : 0,
    'A#': 1, 'B-': 1,
    'B' : 2,
    'C' : 3,
    'C#': 4, 'D-': 4,
    'D' : 5,
    'D#': 6, 'E-': 6,
    'E' : 7,
    'F' : 8,
    'F#': 9, 'G-': 9,
    'G' :10,
    'G#':11, 'A-': 11,
    'R' :12
}

note_reverse_dict = {
    0: 'A',
    1: 'A#',
    2: 'B',
    3: 'C',
    4: 'C#',
    5: 'D',
    6: 'D#',
    7: 'E',
    8: 'F',
    9: 'F#',
    10:'G',
    11:'G#',
    12:'R'
}

vocab_size = len(note_reverse_dict)



    #   rest is dev purposes



class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size



    # Global Helpers #



def pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))



def write_loss(epoch_losses, as_txt=False, epoch_nr=None):
    if not as_txt:
        for i, loss in enumerate(epoch_losses):
            print('{{"metric": "Loss {}", "value": {}}}'.format(i+1, float(loss)))
    else:
        for _, loss in enumerate(epoch_losses):
            with open('loss_'+str(_+1)+'.txt','a+') as f:
                f.write(str(epoch_nr)+','+str(loss)+'\n')

def initialize_loss_txt():
    for _ in range(4):
        open('loss_' + str(_ + 1) + '.txt', "w+").close()



def save_model(model, model_id=None, asText=False):
    model_id = '' if model_id is None else str(model_id)
    try:
        if not asText: pickle_save(model,'model' + model_id + '.pkl')
        else:
            with open('summ_models.txt','a') as file:
                file.write(f"> Epoch : {model_id} Parameters \n")
                for i, layer in enumerate(model):
                    file.write(f"Layer : {i} \n")
                    for key in layer:
                        file.write(key+" "+str(layer[key])+'\n')
                file.write('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
    except: print('Model save error.')

def load_model(model_id=None):
    model_id = '' if model_id is None else str(model_id)
    model = None
    try:
        model = pickle_load('model' + model_id + '.pkl')
        print('> Model loaded.')
    except:
        print('> Model not found.')
    finally: return model



def load_data(data_path, limit_size):
    raw_files = glob.glob(data_path)
    random.shuffle(raw_files)

    data = []
    for file in raw_files:
        dataset = pickle_load(file)

        sample_X, sample_Y = dataset
        vocab_X, oct_X, dur_X, vol_X = sample_X
        vocab_Y, oct_Y, dur_Y, vol_Y = sample_Y

        blocks = []
        for _ in range(len(vocab_X)):
            blocks.append([vocab_X[_], oct_X[_], dur_X[_], vol_X[_],
                           vocab_Y[_], oct_Y[_], dur_Y[_], vol_Y[_]])

        data.extend(blocks) # data.extend(random.choices(blocks, k=limit_size))
        if len(data) > limit_size:

            data = data[:limit_size]
            break

    return data

def get_datasize(data_path):
    files = glob.glob(data_path)
    total_size = 0
    for file in files:
        data = pickle_load(file)
        total_size += len(data[0][0])
    return total_size

def batchify(resource, batch_size):
    data_size = len(resource)
    # todo: do


def plot_loss_txts(hm_mins_refresh=2):
    from matplotlib import style
    import matplotlib.pyplot as plot
    import matplotlib.animation as animation
    # import matplotlib.patches as mpatches
    import random

    loss_1_path = 'loss_1.txt'  # input('import loss_1: ')
    loss_2_path = 'loss_2.txt'  # input('import loss_2: ')
    loss_3_path = 'loss_3.txt'  # input('import loss_3: ')
    loss_4_path = 'loss_4.txt'  # input('import loss_4: ')

    fig = plot.figure()
    axis = fig.add_subplot(1, 1, 1)

    theme = random.choice(['Solarize_Light2', 'fivethirtyeight'])
    style.use(theme)

    def animate(i):
        epochs, losses = [], []

        with open(loss_1_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    epochs.append(int(epoch))
                    losses.append(int(loss))

        with open(loss_2_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))

        with open(loss_3_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))

        with open(loss_4_path, 'r') as f:
            for line in f.readlines():
                epoch, loss = line.split(',')
                loss = float(loss[:-1])
                if loss != 999999999:
                    losses.append(int(loss))

        axis.clear()
        axis.plot(epochs, losses[:len(epochs)], 'r', label='Vocabulary')
        axis.plot(epochs, losses[len(epochs) * 2: len(epochs) * 3], 'b', label='Rhythm')
        axis.plot(epochs, losses[len(epochs):len(epochs) * 2], 'g', label='Octaves')
        axis.plot(epochs, losses[len(epochs) * 3: len(epochs) * 4], 'y', label='Velocities')

        plot.legend(bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=0.)

    ani = animation.FuncAnimation(fig, animate, hm_mins_refresh)
    var = plot.gcf()
    var.canvas.set_window_title('Loss Plot')

    plot.show()





if __name__ == '__main__':
    plot_loss_txts()



# music21 basic guide:
#   for element in stream:
# element.pitch.name
# element.pitch.octave
# element.duration.quarterLength
# element.volume.velocity
