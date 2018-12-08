(All is Optional Usage)



@ Other Menu Options

menu ->
 
-> "debug": model & training & data files recovery and cleanup

-> "graph": plot single or multiple loss(es) 
	(requires Matplotlib)





@ Training Params


	startadv:

run training using adaptive momentum
default: startadv=False


	lr1, lr2:

learning rates for two modes respectively
default: lr1=0.001
default: lr2=0.01


	drop:

neurons randomly "drop" to generalize better
default: drop=0.1


	onlyloss:

model trained on only a single dimension of output
default: onlyloss=None
(suggested way of using is to train initially by onlyloss=1 then onlyloss=2,3,4)


	adv

(not frequently used)
both modes of sgd are run on same dataset, sequentially, no shuffle occurs. 
(normally, each time a training session is run, a random part of samples data is fetched.)
default: adv=False