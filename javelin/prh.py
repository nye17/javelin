#Last-modified: 01 Mar 2012 02:48:46 AM

from zylc import zyLC, get_data
from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from spear import spear
from gp import FullRankCovariance, NearlyFullRankCovariance

""" PRH likelihood calculation.
"""

my_neg_inf = float(-1.0e+300)

class PRH(object):
    """
    Press+Rybicki+Hewitt
    """
    def __init__(self, zylc, set_warning=False):
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
        self.set_warning = set_warning
        # cadence
        self.cont_cad = zylc.cont_cad

    def lnlikefn(self, covfunc=None, rank="Full", retq=False, **covparams):
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
            if self.set_warning :
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

    def __call__(self, p):
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        logp  = self.lnlikefn(sigma=sigma, tau=tau)
        if tau > self.cont_cad :
            prior = - np.log(tau/self.cont_cad) - np.log(sigma)
        else :
            prior = - np.log(self.cont_cad/tau) - np.log(sigma)
        logp = logp + prior
        print(logp)
        return(logp)


class ContinuumDRW(object) :
    def __init__(self, zylc) :
        self.prh = PRH(zylc)
        self.cont_cad = self.prh.cont_cad
    def __call__(self, p):
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        logp  = self.prh.lnlikefn(sigma=sigma, tau=tau)
        if tau > self.cont_cad :
            prior = - np.log(tau/self.cont_cad) - np.log(sigma)
        else :
            prior = - np.log(self.cont_cad/tau) - np.log(sigma)
        logp = logp + prior
        print(logp)
        return(logp)


def func(p, prh):
    return(prh(p))

#makeContextFunctions()




if __name__ == "__main__":    
    import matplotlib.pyplot as plt
    import pickle
    from emcee import *

    sigma, tau = (2.00, 100.0)
    lagy, widy, scaley = (150.0,  3.0, 2.0)
    lagz, widz, scalez = (200.0,  9.0, 0.5)
    lags   = np.array([0.0,   lagy,   lagz])
    wids   = np.array([0.0,   widy,   widz])
    scales = np.array([1.0, scaley, scalez])

    if False :
        lcfile = "dat/loopdeloop_con.dat"
        zylc   = get_data(lcfile)
        prh    = PRH(zylc)
        print(prh.lnlikefn(tau=tau, sigma=sigma))

    if False :
        lcfile = "dat/loopdeloop_con_y_z.dat"
        zylc   = get_data(lcfile)
        prh    = PRH(zylc)
        print(prh.lnlikefn(covfunc="spear", sigma=sigma, tau=tau, 
            lags=lags, wids=wids, scales=scales))

    if False :
        lcfile = "dat/loopdeloop_con.dat"
        zylc   = get_data(lcfile)
#        cont   = ContinuumDRW(zylc)
        cont   = PRH(zylc)
        nwalkers = 100
        p0 = np.random.rand(nwalkers*2).reshape(nwalkers, 2)
        p0[:,0] = np.abs(p0[:,0]) - 0.5
        p0[:,1] = np.abs(p0[:,1]) + 1.0
#        sampler = EnsembleSampler(nwalkers, 2, cont, threads=1)
#        sampler = EnsembleSampler(nwalkers, 2, cont, threads=2)
        sampler = EnsembleSampler(nwalkers, 2, func, args=[cont,], threads=2)


#        sampler = EnsembleSampler(nwalkers, 2, PRH(zylc), threads=2)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        np.savetxt("burn.out", sampler.flatchain)
        print("burn-in finished\n")
        sampler.reset()
        sampler.run_mcmc(pos, 100, rstate0=state)
        af = sampler.acceptance_fraction
        print(af)
        np.savetxt("test.out", sampler.flatchain)
#        plt.hist(np.exp(sampler.flatchain[:,0]), 100)
#        plt.show()
#        plt.hist(np.exp(sampler.flatchain[:,1]), 100)
#        plt.show()


    if False :
        import multiprocessing as mp
        lcfile = "dat/loopdeloop_con.dat"
        pool = mp.Pool()
        zylc   = get_data(lcfile)
        cont   = PRH(zylc)
#        pool.map(func, args=[cont,])
        pool.map(cont)
        pool.close()
        pool.join()


    

