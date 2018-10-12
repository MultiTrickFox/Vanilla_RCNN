import Vanilla
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

filters = (
    Vanilla.default_filters,
    ((3,4),(6,7,8),(9,10,11)),
    ((4,7),(3,7),(7,8,9),(4,7))
)

filters = (*filters[0], *filters[-1])


    # data details

data_path = "samples_*.pkl"
data_size = 10_000 # todo: re-adjust dis ,
batch_size = 400 # /2


    # training details

start_advanced = False

further_parenting = False

trainer.dropout = 0.0

reducing_batch_sizes = True
reduce_batch_per_epoch = 5
reduce_ratio = 3/5

save_intermediate_model = True
save_model_per_epoch = 5

branch_ctr_max = 5


    # global declarations

loss_initial = \
    [[999_999_999,999_999_999,999_999_999,999_999_999]]
trainer.filters = filters


def simple_parenting(model, accugrads, data):


        # initial conditions

    trainer.learning_rate = learning_rate_1

    ctr_save_id = 0

    successful_epochs = 0

    checkpoints = []

    prevStep = (model, accugrads, loss_initial)


        # begin parenting

    print(f'Batch size : {batch_size}')

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
                resources.save_model(prevStep[0], save_id)
                trainer.save_accugrads(prevStep[1], save_id)
                print(f'Data saved : {ctr_save_id}')

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

                if all(np.array(this_loss[0]) <= np.array(prev_loss[0])):

                    branch_points.append(branch_prevStep)

                    if all(np.array(this_loss[0]) <= np.array(branch_goal[0])):

                        prevStep = branch_prevStep

                        checkpoints.append(prevStep)

                        successful_epochs +=1

                        print(f'@ {get_clock()} : '
                              f'.  epoch {successful_epochs} / {total_epochs} completed. ')
                        resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

                        if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                            trainer.batch_size = int(trainer.batch_size * reduce_ratio)

                        if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                            ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                            resources.save_model(branch_prevStep[0], save_id)
                            trainer.save_accugrads(branch_prevStep[1], save_id)

                        break

                    branch_prevStep = branch_thisStep

                branch_ctr += 1


    del checkpoints[0]
    return checkpoints



def advanced_parenting(model, accugrads, moments, data):


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

        if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

            checkpoints.append(prevStep)

            successful_epochs +=1

            print(f'@ {get_clock()} : '
                  f'.  epoch {successful_epochs} / {total_epochs} completed.')
            resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

            if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                trainer.batch_size = int(trainer.batch_size * 4/5)

            if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                resources.save_model(prevStep[0], save_id)
                trainer.save_accugrads(prevStep[1], save_id)
                trainer.save_moments(prevStep[2], save_id)

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

                branch_thisStep = trainer.train_rms(prev_model, prev_accugrads, prev_moments, data) ; this_loss = branch_thisStep[-1]

                if all(np.array(this_loss[0]) < np.array(prev_loss[0])):

                    branch_points.append(branch_prevStep)

                    if all(np.array(this_loss[0]) < np.array(branch_goal[0])):

                        prevStep = branch_prevStep

                        checkpoints.append(prevStep)

                        successful_epochs +=1

                        print(f'@ {get_clock()} : '
                              f'.  epoch {successful_epochs} / {total_epochs} completed. ')
                        resources.write_loss(this_loss[0], as_txt=True, epoch_nr=successful_epochs)

                        if reducing_batch_sizes and successful_epochs % reduce_batch_per_epoch == 0:
                            trainer.batch_size = int(trainer.batch_size * 4/5)

                        if save_intermediate_model and successful_epochs % save_model_per_epoch == 0:
                            ctr_save_id +=1 ; save_id = ctr_save_id * 0.001
                            resources.save_model(branch_prevStep[0], save_id)
                            trainer.save_accugrads(branch_prevStep[1], save_id)
                            trainer.save_moments(branch_prevStep[2], save_id)

                        break

                    branch_prevStep = branch_thisStep

                branch_ctr +=1


    del checkpoints[0]
    return checkpoints



# helpers

def get_data(): return resources.load_data(data_path, data_size)

def get_clock(): return time.asctime(time.localtime(time.time())).split(' ')[3]



# parent runners


def run_simple_parenting(data):

    # initialize model
    model = resources.load_model()
    if model is None:
        model = Vanilla.create_model(filters)
    else:
        resources.save_model(model, '_before_simple')

    # initialize metadata
    accugrads = trainer.load_accugrads(model)

    # get checkpoints
    checkpoints = simple_parenting(model, accugrads, data)

    # extract metadata
    model = checkpoints[-1][0]
    accugrads = checkpoints[-1][1]

    # save metadata
    resources.save_model(model)
    trainer.save_accugrads(accugrads)


def run_advanced_parenting(data):

    # initialize model
    model = resources.load_model()
    if model is None:
        model = Vanilla.create_model(filters)
    else:
        resources.save_model(model, '_before_advanced')

    # initalize metadata
    accugrads = trainer.load_accugrads(model)
    moments = trainer.load_moments(model)

    # get checkpoints
    checkpoints = advanced_parenting(model, accugrads, moments, data)

    # extract metadata
    model = checkpoints[-1][0]
    accugrads = checkpoints[-1][1]
    moments = checkpoints[-1][2]

    # save metadata
    resources.save_model(model)
    trainer.save_accugrads(accugrads)
    trainer.save_moments(moments)



def parent_bootstrap():
    resources.initialize_loss_txt() ; torch.set_default_tensor_type('torch.FloatTensor')

    data = get_data()

    trainer.batch_size = batch_size

    if not start_advanced:  # start simple

        run_simple_parenting(data)

        if further_parenting:  # then further parent

            run_advanced_parenting(data)

    else:  # OR start advanced

        run_advanced_parenting(data)


if __name__ == '__main__':

    parent_bootstrap()
