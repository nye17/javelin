#Last-modified: 16 Jan 2012 05:39:41 PM

from zylc import zyLC
from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from gp import Covariance

""" PRH likelihood calculation.
"""


my_neg_inf = float(-1.0e+300)

class PRH(object):
    """
    """
    def __init__(self, zylc, covfunc, **covparams):
        if not isinstance(zylc, zyLC):
            raise RuntimeError("zylc has to be a zyLC object")
        # initialize zylc
        self.zylc = zylc
        self.npt  = zylc.npt
        self.nlc  = zylc.nlc
        self.jarr = zylc.jarr
        self.marr = zylc.marr.T  # make it a vector
        self.earr = zylc.earr
        self.iarr = zylc.iarr
        self.varr = np.power(zylc.earr, 2.)
        self.larr = np.zeros((self.npt, self.nlc))
        for i in xrange(self.npt):
            lcid = self.iarr[i] - 1
            self.larr[i, lcid] = 1.0
        self.larrTr = self.larr.T

        if (self.nlc == 1):
            self.set_single = True
        else:
            self.set_single = False

        if self.set_single:
            covfunc_dict = get_covfunc_dict(covfunc, **covparams)
            self.C = Covariance(covfunc_dict)
        else:
            pass
            raise RuntimeError("Sorry, RM part not implemented yet")


    def loglike_prh(self, U=None, retq=True):
        # cholesky decompose S+N so that U^T U = S+N = C
        if self.set_single:
            cmatrix = self.C(self.jarr, self.jarr)
            U, info = cholesky(cmatrix, nugget=self.varr, inplace=True, raiseinfo=False)
            if info > 0:
                print("warning: cmatrix non positive-definite")
                if retq:
                    return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf)
                else:
                    return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf)
        elif U is None:
            pass
            raise RuntimeError("require U of cpnmatrix for more RM purposes")

        detC_log = chodet_from_tri(U, retlog=True)
        # solve for C a = y so that a = C^-1 y 
        a = chosolve_from_tri(U, self.marr)
        # solve for C b = L so that b = C^-1 L
        b = chosolve_from_tri(U, self.larr)
        # multiply L^T and b so that C_p = L^T C^-1 L = C_q^-1
        C_p = np.dot(self.larrTr, b)

        # for 'set_single is True' case, C_p is a scalar.
        if self.set_single:
            # for single-mode, cholesky of C_p is simply squre-root of C_p
            W = np.sqrt(C_p)
            detCp_log = np.log(C_p.squeeze())
            # for single-mode, simply devide L^T by C_p
            d = self.larrTr/C_p
        else:
            # cholesky decompose C_p so that W^T W = C_p
            W = cholesky(C_p)
            detCp_log = chodet_from_tri(W, retlog=True)
            # solve for C_p d = L^T so that d = C_p^-1 L^T = C_q L^T
            d = chosolve_from_tri(W, self.larrTr)

        # multiply b d and a so that e = C^-1 L C_p^-1 L^T C^-1 y 
        e = np.dot(b, np.dot(d, a))
        # a minus e so that f = a - e = C^-1 y - C^-1 L C_p^-1 L^T C^-1 y
        #              thus f = C_v^-1 y
        f = a - e
        # multiply y^T  and f so that h = y^T C_v^-1 y
        h = np.dot(self.marr, f)
        # chi2_PRH = -0.5*h
        _chi2 = -0.5*h
        # following Carl Rasmussen's term, a penalty on the complexity of the model
        _compl_pen = -0.5*detC_log
        # penalty on blatant linear drift
        _wmean_pen = -0.5*detCp_log
        # final log_likelhood
        _log_like = _chi2 + _compl_pen + _wmean_pen
        if retq:
            q = np.dot(d, a)
            # q[0] to take the scalar value from the 0-d array
            return(_log_like, _chi2, _compl_pen, _wmean_pen, q[0])
        else:
            return(_log_like, _chi2, _compl_pen, _wmean_pen)




if __name__ == "__main__":    
    import lcio as IO

    lcfile = "mock.dat"
    lclist = IO.readlc_3c(lcfile)

    zylc = zyLC(zylclist=lclist)
    prh = PRH(zylc, tau=200.0, sigma=0.05, nu=0.5)
    print(prh.loglike_prh())
    

