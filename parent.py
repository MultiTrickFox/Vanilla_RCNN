# import Vanilla
import Vanilla_attention as Vanilla

import trainer
import resources

import time

import torch
import numpy as np



    # parent details

total_epochs = 20
learning_rate_1 = 0.001
learning_rate_2 = 0.01


    # model details

filters =                    \
    Vanilla.default_filters +  \
    ()  # +                       \
    #   ((4, 7), (3, 7), (9, 4))  \

layers = (10, 8, 12)


    # data details

data_path = "sample*.pkl"
data_size = 25_000
batch_size = 400


    # training details

start_advanced = False

further_parenting = False

dropout = 0.1

reducing_batch_sizes = False
reduce_batch_per_epoch = 10
reduce_ratio = 95/100

save_intermediate_model = True
save_model_per_epoch = 10

branch_ctr_max = 5

really_random_data = True

only_loss_on = None
loss_multipliers = (1, 1, 1, 1)


    # global declarations

loss_initial = \
    ([999_999_999,999_999_999,999_999_999,999_999_999])





def simple_parenting(model, accugrads, data, last_loss):


        # initial conditions

    trainer.learning_rate = learning_rate_1

    ctr_save_id = 0

    successful_epochs = 0

    checkpoints = []
    prevStep = (model, accugrads, loss_initial)


        # begin parenting

    print(f'\n@ {get_clock()} : Simple Parent running...')

    while successful_epochs < total_epochs:

        prev_model, prev_accugrads, prev_loss = prevStep

        thisStep = trainer.train_rms(prev_model, prev_accugrads, data) ; this_loss = thisStep[-1]

        if all(np.array(this_loss[0]) <= np.array(prev_loss[0])):

            checkpoints.append(prevStep)

            successful_epochs +=1

            print(f'@ {get_clock()} : '
                  f'.  epoch {successful_epochs} / {total_epochs} completed. ')
            resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

            if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                trainer.batch_size = int(trainer.batch_size * reduce_ratio)
                print(f'Batch size reduced : {trainer.batch_size}')

            if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                save_checkpoint(prevStep, save_id)
                print(f'Data saved : Part {ctr_save_id}')

            prevStep = thisStep

        else:

            branch_ctr = 0

            branch_points = []

            branch_prevStep = thisStep

            branch_goal = prevStep[-1]

            while branch_ctr < branch_ctr_max:

                print(f'@ {get_clock()} : '
                      f'... branching {branch_ctr+1} / {branch_ctr_max} . ')

                prev_model, prev_accugrads, prev_loss = branch_prevStep

                branch_thisStep = trainer.train_rms(prev_model, prev_accugrads, data) ; this_loss = branch_thisStep[-1]

                if all(np.array(this_loss[0]) <= np.array(branch_goal[0])) and branch_ctr != 0:

                    checkpoints.append(branch_prevStep)

                    successful_epochs +=(branch_ctr+2)

                    print(f'@ {get_clock()} : '
                          f'.  epoch {successful_epochs} / {total_epochs} completed. ')
                    resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

                    if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                        trainer.batch_size = int(trainer.batch_size * reduce_ratio)

                    if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                        ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                        save_checkpoint(branch_prevStep, save_id)
                        print(f'Data saved : Part {ctr_save_id}')

                    prevStep = branch_thisStep

                    break

                elif all(np.array(this_loss[0]) <= np.array(prev_loss[0])) and branch_ctr != 0:

                    branch_points.append(branch_prevStep)

                    branch_prevStep = branch_thisStep

                branch_ctr +=1


    del checkpoints[0]
    return checkpoints



def advanced_parenting(model, accugrads, moments, data, last_loss):


        # initial conditions

    trainer.learning_rate = learning_rate_2

    ctr_save_id = 0

    successful_epochs = 0

    checkpoints = []

    prevStep = (model, accugrads, moments, loss_initial)


        # begin parenting

    print(f'\n@ {get_clock()} : Advanced Parent running...')

    while successful_epochs < total_epochs:

        prev_model, prev_accugrads, prev_moments, prev_loss = prevStep

        thisStep = trainer.train_adam(prev_model, prev_accugrads, prev_moments, data, epoch_nr=successful_epochs) ; this_loss = thisStep[-1]

        if all(np.array(this_loss[0]) <= np.array(prev_loss[0])):

            checkpoints.append(prevStep)

            successful_epochs +=1

            print(f'@ {get_clock()} : '
                  f'.  epoch {successful_epochs} / {total_epochs} completed.')
            resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

            if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                trainer.batch_size = int(trainer.batch_size * 4/5)

            if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                save_checkpoint(prevStep, save_id)
                print(f'Data saved : Part {ctr_save_id}')

            prevStep = thisStep

        else:

            branch_ctr = 0

            branch_points = []

            branch_prevStep = thisStep

            branch_goal = prevStep[-1]

            while branch_ctr < branch_ctr_max:

                print(f'@ {get_clock()} : '
                      f'... branching {branch_ctr+1} / {branch_ctr_max} . ')

                prev_model, prev_accugrads, prev_moments, prev_loss = branch_prevStep

                branch_thisStep = trainer.train_adam(prev_model, prev_accugrads, prev_moments, data) ; this_loss = branch_thisStep[-1]

                if all(np.array(this_loss[0]) <= np.array(branch_goal[0])) and branch_ctr != 0:

                    checkpoints.append(branch_prevStep)

                    successful_epochs +=(branch_ctr+2)

                    print(f'@ {get_clock()} : '
                          f'.  epoch {successful_epochs} / {total_epochs} completed. ')
                    resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

                    if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                        trainer.batch_size = int(trainer.batch_size * 4/5)

                    if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                        ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                        save_checkpoint(branch_prevStep, save_id)
                        print(f'Data saved : Part {ctr_save_id}')

                    prevStep = branch_thisStep

                    break

                if all(np.array(this_loss[0]) <= np.array(prev_loss[0])) and branch_ctr != 0:

                    branch_points.append(branch_prevStep)

                    branch_prevStep = branch_thisStep

                branch_ctr +=1


    del checkpoints[0]
    return checkpoints



# helpers

def get_data(): return resources.load_data(data_path, data_size,really_random=really_random_data)

def get_clock(): return time.asctime(time.localtime(time.time())).split(' ')[4]

def save_checkpoint(step, save_id=None):
    for _,e in enumerate(step[:-1]):
        if   _ == 0: resources.save_model(e, save_id)
        elif _ == 1: trainer.save_accugrads(e, save_id)
        elif _ == 2: trainer.save_moments(e, save_id)

    save_id = "" if save_id is None else str(save_id)
    resources.pickle_save(step[-1], 'meta'+save_id+'.pkl')

import os
def cleanup_past_moments():
    try:os.remove('model_moments.pkl')
    except:pass



# parent runners


def run_simple_parenting(data):

    # initialize model
    model = resources.load_model()
    if model is None:
        model = Vanilla.create_model(filters)
    else:
        resources.save_model(model, '_before_simple')

    # initialize metadata
    last_loss = resources.pickle_load('meta.pkl')
    if last_loss is None: last_loss = loss_initial
    else: last_loss = [[e if e>0 else 999_999] for e in last_loss[0]]
    accugrads = trainer.load_accugrads(model)

    # get checkpoints
    checkpoints = simple_parenting(model, accugrads, data, last_loss)
    save_checkpoint(checkpoints[-1], save_id=None)

    # # extract metadata
    # model = checkpoints[-1][0]
    # accugrads = checkpoints[-1][1]
    #
    # # save metadata
    # resources.save_model(model)
    # trainer.save_accugrads(accugrads)


def run_advanced_parenting(data):

    # initialize model
    model = resources.load_model()
    if model is None:
        model = Vanilla.create_model(filters)
    else:
        resources.save_model(model, '_before_advanced')

    # initalize metadata
    last_loss = resources.pickle_load('meta.pkl')
    if last_loss is None: last_loss = loss_initial
    else: last_loss = [[e if e>0 else 999_999] for e in last_loss[0]]
    accugrads = trainer.load_accugrads(model)
    moments = trainer.load_moments(model)

    # get checkpoints
    checkpoints = advanced_parenting(model, accugrads, moments, data, last_loss)
    save_checkpoint(checkpoints[-1], save_id=None)

    # # extract metadata
    # model = checkpoints[-1][0]
    # accugrads = checkpoints[-1][1]
    # moments = checkpoints[-1][2]
    #
    # # save metadata
    # resources.save_model(model)
    # trainer.save_accugrads(accugrads)
    # trainer.save_moments(moments)



def bootstrap(custom=False,ep=None,ds=None,bs=None):

    # init internal

    if custom:
        global total_epochs; total_epochs = ep
        global data_size   ; data_size    = ds
        global batch_size  ; batch_size   = bs

    global advanced_parenting, further_parenting
    if advanced_parenting is not None and further_parenting is not None:
        if further_parenting: advanced_parenting = False

    # connections to trainer set

    trainer.layers = layers
    trainer.filters = filters
    trainer.dropout = dropout
    trainer.batch_size = batch_size
    trainer.loss_multipliers = loss_multipliers
    trainer.which_loss = only_loss_on

    # display details

    print(f'Data size  : {data_size}')
    print(f'Batch size : {batch_size}')
    print(f'Epochs     : {total_epochs}')
    print('')

    # init

    torch.set_default_tensor_type('torch.FloatTensor')
    resources.initialize_loss_txt()

    data = get_data()

    # start

    if not start_advanced:     # start simple

        run_simple_parenting(data)

        if further_parenting:  # then further parent

            run_advanced_parenting(data)

        else:cleanup_past_moments()

    else:                      # OR start advanced

        run_advanced_parenting(data)


if __name__ == '__main__':

    bootstrap()
