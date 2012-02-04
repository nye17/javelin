#Last-modified: 03 Feb 2012 07:13:28 PM

import numpy as np


class SpearCovariance(object):
    def __init__(self, eval_fun, **params):
        self.eval_fun = eval_fun
        self.params   = params
    def __call__(self, x, y=None):
        if y is x:
            symm=True
        else:
            symm=False

        if (len(x.shape) != 1):
            if (min(x.shape) == 1):
                x = x.squeeze()
            else:
                raise RuntimeError("MyCovariance can only accept 1d arrays: x")

        lenx = len(x)
        orig_shape = x.shape
        
        if y is None:
            # Special fast-path for functions that have an 'amp' parameter
            if hasattr(self.eval_fun, 'diag_call'):
                V = self.eval_fun.diag_call(x, **self.params)
            # Otherwise, evaluate the diagonal in a loop.
            else:
                V=empty(lenx,dtype=float)
                for i in xrange(lenx):
                    this_x = x[i].reshape((1,-1))
                    V[i] = self.eval_fun(this_x, this_x, **self.params)
            return(V.reshape(orig_shape))
        else:
            # ====================================================
            # = # If x and y are the same array, save some work: =
            # ====================================================
            if symm:
                C=self.eval_fun(x,x,symm=True,**self.params)
                return(C)
            # ======================================
            # = # If x and y are different arrays: =
            # ======================================
            else:
                y = np.atleast_1d(y)
                if (len(y.shape) != 1):
                    if (min(y.shape) == 1):
                        y = y.squeeze()
                    else:
                        raise RuntimeError("MyCovariance can only accept 1d arrays: y")

                C = self.eval_fun(x,y,**self.params)
                return(C)
