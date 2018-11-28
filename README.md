# Vanilla_RCNN
convolution on chord intervals

is a from-scratch RCNN, using only pytorch's autograd

>- between fully connected gru layers, convolutions with incoming chords to come up with a "likely" chord response; while passing information to a gru-lstm stack for deciding details (i.e. pitch, velocity etc.)


requirements:

>python version >= 3.6: https://www.python.org/downloads/release/python-367/ and recommended launcher is IDLE (comes by default with python installation.)

>(built on torch 0.4.0, provided @ (Mac OS X: pip3 install torch=0.4.0 & Windows: pip3 install 
http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-win_amd64.whl - python3.6 version only.)

>also music21 v5.1.0 is required for preprocess 
& interaction purposes @
(pip3 install -Iv music21==5.1.0


>Notice : same versions mentioned above are required & else is known to have bugs.



Guide (simple):


0- Make sure to install packages mentioned above.

>guaranteed to run on OS X, while Windows is known to be memory-error prone.


1- Using provided model:


Responding to Midi Files


>- copy .mid file into project dir


>- Run.py -> Midi Response


MuseScore Interaction 

>- ( requires : https://musescore.org/en. )

>- Run.py -> Interact

>- I/O via MuseScore.


2- (Optional) Start from scratch:

Running on Windows

>- IDLE will be enough.


Running on OS X

>- IDLE can only handle interaction modes and known to crash during training. 

>- Recommend switching to PyCharm ( https://www.jetbrains.com/pycharm/download/ )



Delete the provided model.pkl

>- Manually from project dir

>- Or, run.py -> debug (brings up debug menu)


Custom dataset available on
>https://www.floydhub.com/developersfox/datasets/jazz_piano


Ctrl+C .pkl files into project dir

>.pkls are preprocessed .mid files


(Optional) For training on your own .mid files

>- Create /samples in project dir

>- Ctrl+C .mid files into /samples

>- Run.py -> Preprocess



Have .pkl files ready at project dir

>- Run.py -> Training



(Extra) Training Options:


>- startadv: 

>train with momentum based sgd

>default: startadv=False

>- lr1, lr2: 

>learning rates for sgd and momentum modes

>default: lr1=0.001

>default: lr2=0.01

>- drop: 

>neurons randomly "drop" to generalize better

>default: drop=0.1

>- adv

>(not frequently used)

>sequentially execute basic sgd first, then momentum traing.

>default: adv=False