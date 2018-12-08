# Vanilla_RCNN
convolution on chord intervals

is a from-scratch RCNN, using only pytorch's autograd

> What does it do:

>- parses midi information (or direct i/o) as a template to generate a "sequence of sounds"

>- has chord and solo modes for generating content accordingly

> How does it do:

>- convolutions on incoming chords, between fully connected gru layers, to come up with a "likely" chord response; while passing information to a sub network of gru-gru-lstm stack for deciding details (i.e. pitch, velocity etc.)


requirements:

>python version >= 3.6: https://www.python.org/downloads/release/python-367/

>built on torch 0.4.0, provided @ (Mac OS X: pip3 install torch=0.4.0 & Windows: pip3 install
http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-win_amd64.whl - python3.6 version only.)

>also music21 v5.1.0 is required for preprocess
and interaction purposes @
(pip3 install music21==5.1.0


>Notice : same versions mentioned above are required & else is known to have bugs.


how to run:

>- OS X:

> terminal -> python3 <drag & drop run.py> -> hit enter

>- Windows:

> double click Runner



Guide (simple):


0- Make sure to install packages mentioned above.


1- Using provided model:


Responding to Midi Files


>- copy .mid file into project dir


>- Run.py -> Midi Response


MuseScore Interaction

>- ( requires : https://musescore.org/en. )

>- Run.py -> Interact

>- I/O via MuseScore.


2- (Optional) Start from scratch:



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

>- (Optional) trainer parameters provided as .txt
