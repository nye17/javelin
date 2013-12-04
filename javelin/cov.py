#Last-modified: 04 Dec 2013 16:36:10

all = ['get_covfunc_dict', 'covname_dict', 'MyCovariance']

from gp.cov_funs import matern, pow_exp, pareto_exp, kepler_exp, pow_tail
import numpy as np

""" Wrapping the all the continuum (source, without lines) covariance functions together.
"""

covname_dict = {
                "matern"      : matern.euclidean,
                "pow_exp"     : pow_exp.euclidean,
                "drw"         : pow_exp.euclidean,
                "pareto_exp"  : pareto_exp.euclidean,
                "pow_tail"    : pow_tail.euclidean,
                "kepler_exp"  : kepler_exp.euclidean,
                "kepler2_exp" : kepler_exp.euclidean,
               }


def get_covfunc_dict(covfunc, **covparams):
    """ Try to simplify the procedure of calling different covariance functions
    by calling  all the thrid parameters as *nu*.

    Parameters
    ----------
    covfunc : str
        Names of the covariance funcrtions available, i.e., keys in `covname_dict`, currently including `matern`, `pow_exp`, `drw`, `pareto_exp`, `pow_tail`, `kepler_exp`, and `kepler2_exp`.

    **covparams: 
        keyword arguments for each `convfunc`.

    Returns 
    -------
    _cov_dict : dict
        External namespaces for the name and parameters of `convfunc`.

    """
    _cov_dict = {}
    _cov_dict['eval_fun'] = covname_dict[covfunc]
    _cov_dict['amp']      = covparams['sigma']
    _cov_dict['scale']    = covparams['tau']
    if covfunc == "drw" :
        _cov_dict['pow']         = 1.0
    elif covfunc == "matern" :
        _cov_dict['diff_degree'] = covparams['nu']
    elif covfunc == "pow_exp" : 
        _cov_dict['pow']         = covparams['nu']
    elif covfunc == "pareto_exp" : 
        _cov_dict['alpha']       = covparams['nu']
    elif covfunc == "kepler_exp" : 
        _cov_dict['tcut']        = covparams['nu']
    elif covfunc == "kepler2_exp" : 
        # tcut is the ratio of the cutoff timescale to tau, so if you
        # want nu to be the cutoff timescale....
        _cov_dict['tcut']        = covparams['nu']/covparams['tau']
    elif covfunc == "pow_tail" : 
        _cov_dict['beta']        = covparams['nu']
    else :
        raise RuntimeError("%s has not been implemented"%covfunc)
    return(_cov_dict)


class MyCovariance(object):
    def __init__(self, eval_fun, **params):
        """

        .. note:: Deprecated in Javelin 0.3.
                  `get_covfunc_dict` and `covname_dict` will be used instead.

        """
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
