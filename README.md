# Vanilla_RCNN
convolution on chord intervals

is a from-scratch RCNN, using only pytorch's autograd

>python version >= 3.6 is required: https://www.python.org/downloads/release/python-367/ and recommended launcher is IDLE (comes by default with python installation.)

>(built on torch 0.4.0, provided @ (Mac OS X: pip3 install torch=0.4.0 & Windows: pip3 install 
http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-win_amd64.whl - python3.6 version only.)

>also music21 v5.1.0 is required for preprocess 
& interaction purposes @
(pip3 install -Iv music21==5.1.0


>Notice : versions mentioned above are required & else is known to have bugs.



Guide (simple):


0- Make sure to install packets mentioned above.

>guaranteed to run on OS X, while Windows is known to be memory-error prone.


1- Using provided model:


Responding to midi


>- copy .mid file into project dir


>- run.py -> midi response


Musescore interaction: 

>- I/O via MuseScore. ( Requires : https://musescore.org/en. )

>- run.py -> interact


2- (Optional) Start from scratch:

Delete the provided model.pkl

>- run.py -> debug (brings up debug menu)


Custom dataset available on:
>https://www.floydhub.com/developersfox/datasets/jazz_piano

Ctrl+C .pkl files into project dir




(Optional) for training on custom .mid files, paste as projectdir/samples/your_file.mid

>- (Optional) run.py -> preprocess


Have .pkl files ready at project dir

>- run.py -> training


