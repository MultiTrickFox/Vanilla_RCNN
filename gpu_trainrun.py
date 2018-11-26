import Vanilla
import resources

import torch
import numpy as np

from glob import glob
import random


# lightweight custom torch model gpu trainer


hm_epochs = 20
data_path = "sample*.pkl"
data_size = 30_000
batch_size = 200


def bootstrap():

    if False: # not torch.cuda.is_available():
        input("gpu runner is unavailable on this system.")
    else:

        # torch.set_default_tensor_type('torch.cuda.FloatTensor')


        model = Model()
        data = load_data(data_path, data_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


        for epoch in range(hm_epochs):

            epoch_loss = np.zeros(Vanilla.hm_vectors)

            random.shuffle(data)

            for batch in batchify(data, batch_size):    # todo: batch is 2 element arr -> stacked inputs & stacked targets

                _, loss = model(batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.numpy()



def forward_prop(model, batch, dropout=0.1):


    inputs, targets = batch





    # @ todo: actually write..




    # todo: return a loss as well
    pass


def forward_prop_t(model, inputs_t, targets_t):

    # todo : actually do

    pass





class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                model = resources.load_model()
                if model is None:
                    model = Vanilla.create_model()
                self.model = model

            def forward(self, batch):
                return forward_prop(self.model, batch)


# def saver(model):                                 # optimizer saver & loader as well
#     torch.save(model.state_dict(), "modelsave")
#
# def loader():
#     model = Model()
#     try:
#         model.load_state_dict(torch.load("modelsave"))
#         model.eval()
#     except:
#         print("saved model not found.")
#     return model

def saver(model):
    resources.save_model(model.model)


def load_data(data_path, data_size):

    files = glob(data_path)
    random.shuffle(files)

    data = []

    for file in files:

        dataset = resources.pickle_load(file)
        samples_X, samples_Y = dataset
        vocab_X, oct_X, dur_X, vol_X = samples_X
        vocab_Y, oct_Y, dur_Y, vol_Y = samples_Y

        blocks = []
        for _ in range(len(vocab_X)):
            blocks.append([vocab_X[_], oct_X[_], dur_X[_], vol_X[_],
                           vocab_Y[_], oct_Y[_], dur_Y[_], vol_Y[_]])

        data.extend(blocks) # data.extend(random.choices(blocks, k=limit_size))
        if len(data) > data_size:

            data = data[:data_size]
            break

    return data


    return


def batchify(data, batchsize):

    for datapoint in data:
        x_vocab, x_oct, x_dur, x_vol, y_vocab, y_oct, y_dur, y_vol = datapoint

    # todo : return





if __name__ == '__main__':
    bootstrap()
