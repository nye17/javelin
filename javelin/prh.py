#Last-modified: 29 Feb 2012 06:19:46 PM

from zylc import zyLC, get_data
from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from spear import spear
from gp import FullRankCovariance, NearlyFullRankCovariance
from emcee import *

""" PRH likelihood calculation.
"""

my_neg_inf = float(-1.0e+300)

class PRH(object):
    """
    PressRybickiHewitt
    """
    def __init__(self, zylc):
        """ PRH (initials of W.H. Press, G.B. Rybicki, and J.N. Hewitt) object.

        Parameters
        ----------
        zylc: zyLC object
            Light curve data.
        """
        if not isinstance(zylc, zyLC):
            raise RuntimeError("zylc has to be a zyLC object")
        # initialize zylc
        self.zylc = zylc
        self.nlc  = zylc.nlc
        self.npt  = zylc.npt
        self.jarr = zylc.jarr
        self.marr = zylc.marr.T  # make it a vector
        self.earr = zylc.earr
        self.iarr = zylc.iarr
        self.varr = np.power(zylc.earr, 2.)
        # construct the linear response matrix
        self.larr = np.zeros((self.npt, self.nlc))
        for i in xrange(self.npt):
            lcid = self.iarr[i] - 1
            self.larr[i, lcid] = 1.0
        self.larrTr = self.larr.T
        # dimension of the problem
        self.set_single = zylc.issingle

    def lnlikefn(self, covfunc=None, rank="Full", retq=True, **covparams):
        """ PRH log-likelihood function
        
        Parameters
        ----------
        covfunc: string, optional
            Name of the covariance function (default: None, i.e., use 'drw' for
            single light curve, and 'spear' for multiple light curves ).

        rank: string, optional
            Rank of the covariance function, could potentially use 'NearlyFull'
            rank covariance when the off-diagonal terms become strong (default:
            'Full').

        retq: bool, optional
            Return the value(s) of q along with each component of the
            log-likelihood if True (default: True).

        covparams: kwargs
            Parameters for 'covfunc'. For 'spear', they are positional
            arguments: sigma,tau,lags,wids,scales.

        Returns
        -------
        log_like: float
            Log likelihood.

        chi2: float
            Chi^2 component (setq=True).

        compl_pen: float
            Det component (setq=True).

        wmean_pen: float
            Var(q) component (setq=True).

        q: array_like
            Minimum variance estimate of qs, note that for single light curve
            fitting, q is then an one-element array (setq=True).

        """
        # set up covariance function
        if self.set_single:
            if covfunc is None :
                covfunc = 'drw'
            covfunc_dict = get_covfunc_dict(covfunc, **covparams)
            if rank is "Full" :
                # using full-rank
                C = FullRankCovariance(**covfunc_dict)
            elif rank is "NearlyFull" :
                # using nearly full-rank
                C = NearlyFullRankCovariance(**covfunc_dict)
        else:
            if ((covfunc is None) or (covfunc == "spear")) :
                C = spear(self.jarr,self.jarr,self.iarr,self.iarr, **covparams)
#                print(self.jarr[:5])
#                print(self.iarr[:5])
#                print(C[:5,:5])
            else :
                raise RuntimeError("unknown covfunc name %s"%covfunc)

        # cholesky decompose S+N so that U^T U = S+N = C
        if self.set_single :
            # using intrinsic method of C without explicitly writing out cmatrix
            try :
                U = C.cholesky(self.jarr, observed=False, nugget=self.varr)
                info = 0
            except :
                info = 1
        else :
            U, info = cholesky(C, nugget=self.varr, inplace=True, raiseinfo=False)

        if info > 0 :
            print("Warning: non positive-definite covariance")
            if retq:
                return(my_neg_inf, my_neg_inf, my_neg_inf, 
                        my_neg_inf, my_neg_inf)
            else:
                return(my_neg_inf)

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
#            return(_log_like, _chi2, _compl_pen, _wmean_pen, q[0])
            return(_log_like, _chi2, _compl_pen, _wmean_pen, q)
        else:
            return(_log_like)




if __name__ == "__main__":    
    sigma, tau = (2.00, 100.0)
    lagy, widy, scaley = (150.0,  3.0, 2.0)
    lagz, widz, scalez = (200.0,  9.0, 0.5)
    lags   = np.array([0.0,   lagy,   lagz])
    wids   = np.array([0.0,   widy,   widz])
    scales = np.array([1.0, scaley, scalez])

    if True :
        lcfile = "dat/loopdeloop_con.dat"
        zylc   = get_data(lcfile)
        prh    = PRH(zylc)
        print(prh.lnlikefn(tau=tau, sigma=sigma))

    if True :
        lcfile = "dat/loopdeloop_con_y_z.dat"
        zylc   = get_data(lcfile)
        prh    = PRH(zylc)
        print(prh.lnlikefn(covfunc="spear", sigma=sigma, tau=tau, lags=lags, wids=wids, scales=scales))
    

