#Last-modified: 16 Jan 2012 08:25:04 PM

all = ['get_covfunc_dict', 'covname_dict']

from gp.cov_funs import matern, pow_exp, pareto_exp, kepler_exp

"""
Wrapping the all the covariance functions together.
"""

covname_dict = {
                "matern"    :    matern.euclidean,
                "pow_exp"   :   pow_exp.euclidean,
                "drw"       :   pow_exp.euclidean,
                "pareto_exp":pareto_exp.euclidean,
                "kepler_exp":kepler_exp.euclidean,
               }


def get_covfunc_dict(covfunc, **covparams):
    """ try to simplify the procedure of calling different covariance functions
    by unifying the thrid parameter as *nu*.
    """
    _cov_dict = {}
    _cov_dict['eval_fun'] = covname_dict[covfunc]
    _cov_dict['amp']      = covparams['sigma']
    _cov_dict['scale']    = covparams['tau']
    if   covfunc is "drw" :
        _cov_dict['pow']         = 1.0
    elif covfunc is "matern" :
        _cov_dict['diff_degree'] = covparams['nu']
    elif covfunc is "pow_exp" : 
        _cov_dict['pow']         = covparams['nu']
    elif covfunc is "pareto_exp" : 
        _cov_dict['alpha']       = covparams['nu']
    elif covfunc is "kepler_exp" : 
        _cov_dict['tcut']        = covparams['nu']
    else :
        print("covfuncs currently implemented:")
        print(" ".join(covfunc_dict.keys))
        raise RuntimeError("%s has not been implemented"%covfunc)
    return(_cov_dict)
