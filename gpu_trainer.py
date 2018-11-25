import Vanilla
import torch


#   @ todo: lightweight internal torch model gpu trainer


# training params here

if torch.cuda.is_available():
    input("no gpu on system.")
else:
    pass

    # convert vanilla to internal model

    # forwprop & weight upd
