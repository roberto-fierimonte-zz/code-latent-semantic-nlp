import theano.tensor as T

from theano.printing import Print


def last_d_softmax(x):

    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)

    return out


def theano_print_shape(var, msg):
    """
    Helper function for printing the shape of a Theano expression during run
    time of the Theano graph.

    Parameters
    ----------

    var : Theano expression
        The variable whose shape to be printed at runtime.

    msg : str
        The message to be printed together with the shape.

    Returns
    -------
    A Theano expression which should be used instead of the original expression
    in order the printing to happen.
    """
    if var.ndim == 0:
        pr = Print(msg + "(SCALAR)")(var)
        return T.switch(T.lt(0, 1), var, pr)
    else:
        pr = Print(msg)(T.shape(var))
        return T.switch(T.lt(0, 1), var, T.cast(pr[0], var.dtype))


def theano_print_value(var, msg):

    if var.ndim == 0:
        pr = Print(msg + "(SCALAR)")(var)
        return T.switch(T.lt(0, 1), var, pr)
    else:
        pr = Print(msg)(var)
        return T.switch(T.lt(0, 1), var, T.cast(pr[0], var.dtype))