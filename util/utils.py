import numpy as np

import theano
import theano.tensor as T


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    Parameters
    ----------
    arr : array_like
        The data to be converted.
    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def shared_empty(dim=2, dtype=None):
    """Creates empty Theano shared variable.
    Shortcut to create an empty Theano shared variable with
    the specified number of dimensions.
    Parameters
    ----------
    dim : int, optional
        The number of dimensions for the empty variable, defaults to 2.
    dtype : a numpy data-type, optional
        The desired dtype for the variable. Defaults to the Theano
        ``floatX`` dtype.
    Returns
    -------
    Theano shared variable
        An empty Theano shared variable of dtype ``dtype`` with
        `dim` dimensions.
    """
    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))
