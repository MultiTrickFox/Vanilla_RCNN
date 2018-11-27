# Vanilla_RCNN
convolution on chord intervals


is a from-scratch RCNN, using only pytorch's autograd 

>(built on torch 0.4.0, provided https://github.com/developersfox/Pytorch-0.4-_custom -> 
run cpu.bat)
also music21 v5.0.1 is required for preproc 
(not needed besides preprocessing.)
(pip3 install -Iv music21==5.0.1


>Notice : versions mentioned above are required & else is known to have bugs.


How To Use:


1- Start from scratch:

Custom dataset available on:
>https://www.floydhub.com/developersfox/datasets/jazz_piano

Ctrl+C .mid files into /samples,
from Runner:
  >Preprocess;
  >Training
  
 
2- Or use provided model:

>from runner.py select interact.
