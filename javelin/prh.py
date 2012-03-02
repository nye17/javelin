#Last-modified: 01 Mar 2012 09:23:38 PM

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


    def lnlikefn_single(self, covfunc="drw", rank="Full", retq=False,
            **covparams) :
        """ PRH log-likelihood function for the single continuum variability
        fit. 
        
        The reason I separate this from the spear likelihood is that the
        single line fit may call functions to do incomplete Cholesky decomposition
        for some covariance models, in which case the MCMC chains are ralatively
        hard to parallel through simple multiprocessing.

        Parameters
        ----------
        covfunc: string, optional
            Name of the covariance function (default: 'drw' for
            single light curve, and 'spear' for multiple light curves ).

        rank: string, optional
            Rank of the covariance function, could potentially use 'NearlyFull'
            rank covariance when the off-diagonal terms become strong (default:
            'Full').

        retq: bool, optional
            Return the value(s) of q along with each component of the
            log-likelihood if True (default: False).

        covparams: kwargs
            Parameters for 'covfunc'. 

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

        q: float
            Minimum variance estimate of 'q'.

        """
        if not self.set_single:
            raise RuntimeError("lnlikefn_single only works for single mode")

        # set up covariance function
        covfunc_dict = get_covfunc_dict(covfunc, **covparams)
        if rank is "Full" :
            # using full-rank
            C = FullRankCovariance(**covfunc_dict)
        elif rank is "NearlyFull" :
            # using nearly full-rank
            C = NearlyFullRankCovariance(**covfunc_dict)
        else :
            raise RuntimeError("No such option for rank %s"%rank)

        # cholesky decompose S+N so that U^T U = S+N = C
        # using intrinsic method of C without explicitly writing out cmatrix
        try :
            U = C.cholesky(self.jarr, observed=False, nugget=self.varr)
        except :
            return(self._exit_with_retval(retq, 
                   errmsg="Warning: non positive-definite covariance C", 
                   set_verbose=self.set_warning))

        retval = self._lnlike_from_U(self, U, setq=setq)
        return(retval)


    def lnlikefn_spear(self, sigma, tau, lags, wids, scales, retq=False) :
        """ PRH log-likelihood function for the reverbeartion mapping model.

        Parameters
        ----------
        sigma: float
            DRW variability amplitude (not sigmahat).

        tau: float
            DRW time scale.

        lags: array_like
            Lags of each light curves, lags[0] is default to 0.0 for continuum.

        wids: array_like
            Widths of each transfer function, wids[0] is default to 0.0 for
            continuum.

        scales: array_like
            Scales of each transfer function, scales[0] is default to 1.0 for
            continuum.

        retq: bool, optional
            Return the values of q in a list 'qlist' along with each component of the
            log-likelihood if True (default: False).

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

        qlist: list
            Minimum variance estimates of 'q's.

        """
        if self.set_single:
            raise RuntimeError("lnlikefn_spear does not work for single mode")

        C = spear(self.jarr,self.jarr,self.iarr,self.iarr, sigma, tau, lags,
                wids, scales)
#test        C = np.diag(np.ones(self.npt))

        U, info = cholesky(C, nugget=self.varr, inplace=True, raiseinfo=False)

        if info > 0 :
           return(self._exit_with_retval(retq, 
                  errmsg="Warning: non positive-definite covariance C", 
                  set_verbose=self.set_warning))

        retval = self._lnlike_from_U(U, retq=retq)
        return(retval)


    def _lnlike_from_U(self, U, retq=False):
        """ calculate the log-likelihoods from the upper triangle of cholesky
        decomposition.
        """
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
            W, info = cholesky(C_p, raiseinfo=False)
            if info > 0 :
                return(self._exit_with_retval(retq, 
                    errmsg="Warning: non positive-definite covariance W", 
                    set_verbose=self.set_warning))
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
            if self.set_single:
                # q[0] to take the scalar value from the 0-d array
                return(_log_like, _chi2, _compl_pen, _wmean_pen, q[0])
            else :
                return(_log_like, _chi2, _compl_pen, _wmean_pen, q)
        else:
            return(_log_like)

    def _exit_with_retval(self, retq, errmsg=None, set_verbose=False):
        """ Return failure elegantly.
        
        When you are desperate and just want to leave the calculation with appropriate return values
        that quietly speak out your angst.
        """
        if errmsg is not None:
            if set_verbose:
                print("Exit: %s"%errmsg)
        if retq:
            if self.issingle :
                return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf)
            else :
                return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf, [my_neg_inf]*self.nlc)
        else:
            return(my_neg_inf)

class DRW_Model(object) :
    def __init__(self, zylc) :
        self.prh = PRH(zylc)
        self.cont_cad = self.prh.cont_cad
    def __call__(self, p, set_prior=True, rank="Full"):
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        logl  = self.prh.lnlikefn_single(covfunc="drw", sigma=sigma, tau=tau,
                rank=rank, retq=False)
        if set_prior :
            if tau > self.cont_cad :
                prior = - np.log(tau/self.cont_cad) - np.log(sigma)
            else :
                prior = - np.log(self.cont_cad/tau) - np.log(sigma)
        else :
            prior = 0.0
        logp = logl + prior
        return(logp)

class Rmap_Model(object) :
    def __init__(self, zylc) :
        self.prh = PRH(zylc)
        self.cont_cad = self.prh.cont_cad
        self.nlc = self.prh.nlc
    def __call__(self, p, set_prior=True):
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        # assemble lags/wids/scales
        lags   = [0.0, ]
        wids   = [0.0, ]
        scales = [1.0, ]
        for i in xrange(1, self.nlc) : 
            lags.append(p[2+(i-1)*3])
            wids.append(p[3+(i-1)*3])
            scales.append(p[4+(i-1)*3])

        logl  = self.prh.lnlikefn_spear(sigma, tau, lags, wids, scales, retq=False)
        if set_prior :
            if tau > self.cont_cad :
                prior = - np.log(tau/self.cont_cad) - np.log(sigma)
            else :
                prior = - np.log(self.cont_cad/tau) - np.log(sigma)
        else :
            prior = 0.0
        logp = logl + prior
        return(logp)





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
        cont   = DRW_Model(zylc)
        nwalkers = 100
        ndim   = 2
        p0 = np.random.rand(nwalkers*2).reshape(nwalkers, 2)
        p0[:,0] = p0[:,0] - 0.5
        p0[:,1] = p0[:,1] + 1.0
        # cannot use multiprocessing in Continuum models
        sampler = EnsembleSampler(nwalkers, 2, cont, threads=1)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        np.savetxt("burn_drw.out", sampler.flatchain)
        print("burn-in finished\n")
        sampler.reset()
        sampler.run_mcmc(pos, 100, rstate0=state)
        af = sampler.acceptance_fraction
        print(af)
        np.savetxt("test_drw.out", sampler.flatchain)
        plt.hist(np.exp(sampler.flatchain[:,0]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,1]), 100)
        plt.show()

    if True :
        lcfile = "dat/loopdeloop_con_y.dat"
        zylc   = get_data(lcfile)
        rmap   = Rmap_Model(zylc)
        nwalkers = 100
        ndim   = 5
        p0 = np.random.rand(nwalkers*ndim).reshape(nwalkers, ndim)
        p0[:,0] = p0[:,0] - 0.5
        p0[:,1] = p0[:,1] + 1.0
        p0[:,2] = p0[:,2] + 1.2
        p0[:,3] = p0[:,3] + 0.6
        p0[:,4] = p0[:,4] + 0.5
        sampler = EnsembleSampler(nwalkers, ndim, rmap, threads=1)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        np.savetxt("burn_top.out", sampler.flatchain)
        print("burn-in finished\n")
        sampler.reset()
        sampler.run_mcmc(pos, 100, rstate0=state)
        af = sampler.acceptance_fraction
        print(af)
        np.savetxt("test_top.out", sampler.flatchain)
        plt.hist(np.exp(sampler.flatchain[:,0]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,1]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,2]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,3]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,4]), 100)
        plt.show()

    if False :
        lcfile = "dat/loopdeloop_con_y_z.dat"
        zylc   = get_data(lcfile)
        rmap   = Rmap_Model(zylc)
        nwalkers = 100
        ndim   = 8
        p0 = np.random.rand(nwalkers*ndim).reshape(nwalkers, ndim)
        p0[:,0] = p0[:,0] - 0.5
        p0[:,1] = p0[:,1] + 1.0
        p0[:,2] = p0[:,2] + 1.2
        p0[:,3] = p0[:,3] + 0.6
        p0[:,4] = p0[:,4] + 0.5
        p0[:,5] = p0[:,5] + 1.2
        p0[:,6] = p0[:,6] + 0.6
        p0[:,7] = p0[:,7] + 0.5
        sampler = EnsembleSampler(nwalkers, ndim, rmap, threads=1)
        pos, prob, state = sampler.run_mcmc(p0, 100)
        np.savetxt("burn_dou.out", sampler.flatchain)
        print("burn-in finished\n")
        sampler.reset()
        sampler.run_mcmc(pos, 100, rstate0=state)
        af = sampler.acceptance_fraction
        print(af)
        np.savetxt("test_dou.out", sampler.flatchain)
        plt.hist(np.exp(sampler.flatchain[:,0]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,1]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,2]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,3]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,4]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,5]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,6]), 100)
        plt.show()
        plt.hist(np.exp(sampler.flatchain[:,7]), 100)
        plt.show()




