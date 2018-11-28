# Vanilla_RCNN
convolution on chord intervals

is a from-scratch RCNN, using only pytorch's autograd 

>(built on torch 0.4.0, provided https://github.com/developersfox/Pytorch-0.4-_custom -> 
run cpu.bat)
also music21 v5.0.1 is required for preprocess 
& interaction purposes
(pip3 install -Iv music21==5.0.1


>Notice : versions mentioned above are required & else is known to have bugs.



How To Use:


1- With provided model:


Responding to midi


>- copy .mid file into project dir


>- run.py -> midi response


Musescore interaction: 

>- I/O via MuseScore. ( Requires : https://musescore.org/en. )

>- run.py -> interact


2- (Optional) Start from scratch:

Delete the provided model.pkl

Custom dataset available on:
>https://www.floydhub.com/developersfox/datasets/jazz_piano

Ctrl+C .pkl files into project dir




(Optional) for training on custom .mid files, paste as projectdir/samples/your_file.mid

>- (Optional) run.py -> preprocess


Have .pkl files ready at project dir

>- run.py -> training


