from lasagne.nonlinearities import elu
from lasagne.nonlinearities import softplus


def elu_plus_one(x):

    return elu(x) + 1. + 1.e-5
