from lasagne.nonlinearities import elu, softplus, sigmoid

def elu_plus_one(x):

    return elu(x) + 1. + 1.e-5
