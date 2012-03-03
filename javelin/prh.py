#Last-modified: 02 Mar 2012 07:31:02 PM

from zylc import zyLC, get_data
from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from spear import spear
from gp import FullRankCovariance, NearlyFullRankCovariance
from scipy.optimize import fmin
import matplotlib.pyplot as plt

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
        self.names= zylc.names
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
            Chi^2 component (retq=True).

        compl_pen: float
            Det component (retq=True).

        wmean_pen: float
            Var(q) component (retq=True).

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

        retval = self._lnlike_from_U(U, retq=retq)
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
            Chi^2 component (retq=True).

        compl_pen: float
            Det component (retq=True).

        wmean_pen: float
            Var(q) component (retq=True).

        qlist: list
            Minimum variance estimates of 'q's.

        """
        if self.set_single:
            raise RuntimeError("lnlikefn_spear does not work for single mode")

        if sigma<=0.0 or tau<=0.0 or np.min(wids)<0.0 or np.min(scales)<=0.0 :
           return(self._exit_with_retval(retq, 
                  errmsg="Warning: illegal input of parameters", 
                  set_verbose=self.set_warning))

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
    def __init__(self, zylc=None) :
        if zylc is None :
            pass
        else :
            self.prh = PRH(zylc)
            self.cont_cad = self.prh.cont_cad
            self.nlc = self.prh.nlc
        # number of parameters
        self.ndim = 2
        self.vars = ["sigma", "tau"]
        self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]

    def __call__(self, p, set_prior=True, rank="Full"):
        if not hasattr(self, "prh") :
            print("Warning: no PRH object found, no __call__ can be done")
            return("__call__ error ")
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        logl  = self.prh.lnlikefn_single(covfunc="drw", sigma=sigma, tau=tau,
                rank=rank, retq=False)
        if set_prior :
            if tau > self.cont_cad :
                prior = - np.log(tau/self.cont_cad) - np.log(sigma)
            elif tau < 0.00001 :
                prior = my_neg_inf
            else :
                prior = - np.log(self.cont_cad/tau) - np.log(sigma)
        else :
            prior = 0.0
        logp = logl + prior
        return(logp)

    def do_map(self, p_ini, fixed=None, set_prior=True, rank="Full", 
            set_verbose=True):
        p_ini = np.asarray(p_ini)
        if fixed is not None :
            fixed = np.asarray(fixed)
            func = lambda _p : -self.__call__(_p*fixed+p_ini*(1.-fixed),
                    set_prior=set_prior, rank=rank)
        else :
            func = lambda _p : -self.__call__(_p,
                    set_prior=set_prior, rank=rank)
        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if set_verbose :
            print("best-fit parameters are")
            print("sigma %8.3f tau %8.3f"%tuple(np.exp(p_bst)))
            print("with logp  %10.5g "%-v_bst)
        return(-v_bst, p_bst)

    def do_mcmc(self, nwalkers=100, nburn=100, nchain=100, fburn=None,
            fchain=None, set_verbose=True):
        if not hasattr(self, "prh") :
            print("Warning: no PRH object found, no mcmc can be done")
            return(1)
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        if set_verbose :
            print("start burn-in")
            print("nburn = %d nwalkers = %d -> number of burn-in iteration = %d"%
                (nburn, nwalkers, nburn*nwalkers))
        sampler = EnsembleSampler(nwalkers, self.ndim, self.__call__, threads=1)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if fburn is not None :
            if set_verbose :
                print("save burn-in chains to %s"%fburn)
            np.savetxt(fburn, sampler.flatchain)
        if set_verbose :
            print("burn-in finished")
        sampler.reset()
        if set_verbose :
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose :
            print("sampling finished")
        af = sampler.acceptance_fraction
        if set_verbose :
            print("acceptance fractions are\n")
        print(af)
        if fchain is not None :
            if set_verbose :
                print("save MCMC chains to %s"%fchain)
            np.savetxt(fchain, sampler.flatchain)
        # make chain an attritue
        self.flatchain = sampler.flatchain
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

    def get_hpd(self, set_verbose=True):
        hpd = np.zeros((3, self.ndim))
        chain_len = self.flatchain.shape[0]
        pct1sig = chain_len*np.array([0.16, 0.50, 0.84])
        medlowhig = pct1sig.astype(np.int32)
        for i in xrange(self.ndim):
            vsort = np.sort(self.flatchain[:,i])
            hpd[:,i] = vsort[medlowhig]
            if set_verbose :
                print("HPD of %s"%self.vars[i])
                print("low: %8.3f med %8.3f hig %8.3f"%tuple(np.exp(hpd[:,i])))
        # register hpd to attr
        self.hpd = hpd

    def show_hist(self):
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        for i in xrange(self.ndim) :
            plt.hist(self.flatchain[:,i]/ln10, 100)
            plt.xlabel(self.texs[i])
            plt.ylabel("N")
            plt.show()

    def load_chain(self, fchain, set_verbose=True):
        if set_verbose :
            print("load MCMC chain from %s"%fchain)
        self.flatchain = np.genfromtxt(fchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        

class Rmap_Model(object) :
    def __init__(self, zylc=None) :
        if zylc is None :
            pass
        else :
            self.prh = PRH(zylc)
            self.cont_cad = self.prh.cont_cad
            self.nlc = self.prh.nlc
            # number of parameters
            self.ndim = 2 + (self.nlc-1)*3
            self.names = self.prh.names
            self.vars = ["sigma", "tau"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            for i in xrange(1, self.nlc) :
                self.vars.append("_".join(["lag",   self.names[i]]))
                self.vars.append("_".join(["wid",   self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append( "".join([r"$t_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$w_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$s_{", self.names[i] ,r"}$"]))

    def __call__(self, p, conthpd):
        if not hasattr(self, "prh") :
            print("Warning: no PRH object found, no __call__ can be done")
            return("__call__ error ")
        sigma = np.exp(p[0])
        tau   = np.exp(p[1])
        # assemble lags/wids/scales
        lags   = np.zeros(self.nlc)
        wids   = np.zeros(self.nlc)
        scales =  np.ones(self.nlc)
        for i in xrange(1, self.nlc) : 
            lags[i]   = p[2+(i-1)*3]
            wids[i]   = p[3+(i-1)*3]
            scales[i] = p[4+(i-1)*3]
        logl = self.prh.lnlikefn_spear(sigma, tau, lags, wids, scales, retq=False)
        # conthpd is in natural log
        # for sigma
        if p[0] < conthpd[1,0] :
            prior0 = (p[0] - conthpd[1,0])/(conthpd[1,0]-conthpd[0,0])
        else :
            prior0 = (p[0] - conthpd[1,0])/(conthpd[2,0]-conthpd[1,0])
        # for tau
        if p[1] < conthpd[1,1] :
            prior1 = (p[1] - conthpd[1,1])/(conthpd[1,1]-conthpd[0,1])
        else :
            prior1 = (p[1] - conthpd[1,1])/(conthpd[2,1]-conthpd[1,1])
        prior = -0.5*(prior0*prior0+prior1*prior1)
        logp = logl + prior
        return(logp)

    def do_map(self, p_ini, fixed=None, conthpd=None, set_verbose=True):
        p_ini = np.asarray(p_ini)
        if fixed is not None :
            fixed = np.asarray(fixed)
            func = lambda _p : -self.__call__(_p*fixed+p_ini*(1.-fixed),
                    conthpd=conthpd)
        else :
            func = lambda _p : -self.__call__(_p, conthpd=conthpd)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if set_verbose :
            print("best-fit parameters are")
            print("sigma %8.3f tau %8.3f"%tuple(np.exp(p_bst[0:2])))
            for i in xrange(1,self.nlc) :
                ip = 2+(i-1)*3
                print("%s %8.3f %s %8.3f %s %8.3f"%(
                    self.vars[ip+0], p_bst[ip+0],
                    self.vars[ip+1], p_bst[ip+1],
                    self.vars[ip+2], p_bst[ip+2],
                    ))
            print("with logp  %10.5g "%-v_bst)
        return(-v_bst, p_bst)

    def do_mcmc(self, conthpd, lags_ini=None, nwalkers=100, nburn=100, nchain=100,
            fburn=None, fchain=None, set_verbose=True):
        if not hasattr(self, "prh") :
            print("Warning: no PRH object found, no mcmc can be done")
            return(1)
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        p0[:, 0] += conthpd[1,0]-0.5
        p0[:, 1] += conthpd[1,1]-0.5
        if lags_ini is not None :
            for i in xrange(1, self.nlc) :
                # lags_ini also starts with continuum lag, which is by
                # definition 0.0
                p0[:, 2+(i-1)*3] += lags_ini[i]-0.499
        if set_verbose :
            print("start burn-in")
            print("using priors on sigma and tau from the continuum fitting")
            print(np.exp(conthpd))
            if lags_ini is not None :
                print("using initial lags as")
                print(lags_ini)
            print("nburn = %d nwalkers = %d -> number of burn-in iteration = %d"%
                (nburn, nwalkers, nburn*nwalkers))
        sampler = EnsembleSampler(nwalkers, self.ndim, self.__call__,
                args=(conthpd,), threads=1)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if fburn is not None :
            if set_verbose :
                print("save burn-in chains to %s"%fburn)
            np.savetxt(fburn, sampler.flatchain)
        if set_verbose :
            print("burn-in finished")
        sampler.reset()
        if set_verbose :
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose :
            print("sampling finished")
        af = sampler.acceptance_fraction
        if set_verbose :
            print("acceptance fractions are\n")
        print(af)
        if fchain is not None :
            if set_verbose :
                print("save MCMC chains to %s"%fchain)
            np.savetxt(fchain, sampler.flatchain)
        # make chain an attritue
        self.flatchain = sampler.flatchain
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

    def get_hpd(self, set_verbose=True):
        hpd = np.zeros((3, self.ndim))
        chain_len = self.flatchain.shape[0]
        pct1sig = chain_len*np.array([0.16, 0.50, 0.84])
        medlowhig = pct1sig.astype(np.int32)
        for i in xrange(self.ndim):
            vsort = np.sort(self.flatchain[:,i])
            hpd[:,i] = vsort[medlowhig]
            if set_verbose :
                print("HPD of %s"%self.vars[i])
                if i < 2 :
                    print("low: %8.3f med %8.3f hig %8.3f"%tuple(np.exp(hpd[:,i])))
                else :
                    print("low: %8.3f med %8.3f hig %8.3f"%tuple(hpd[:,i]))
        # register hpd to attr
        self.hpd = hpd

    def show_hist(self):
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        for i in xrange(self.ndim) :
            if i < 2 :
                plt.hist(self.flatchain[:,i]/ln10, 100)
            else : 
                plt.hist(self.flatchain[:,i], 100)
            plt.xlabel(self.texs[i])
            plt.ylabel("N")
            plt.show()

    def load_chain(self, fchain, set_verbose=True):
        if set_verbose :
            print("load MCMC chain from %s"%fchain)
        self.flatchain = np.genfromtxt(fchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)




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
#        cont.do_mcmc(nwalkers=100, nburn=50, nchain=50, 
#                fburn="burn.dat", fchain="chain.dat")
        p_ini = [0.0, 1.0]
        cont.do_map(p_ini, fixed=None, set_prior=False, rank="Full", 
            set_verbose=True)

    if True :
        lcfile = "dat/loopdeloop_con_y.dat"
        zylc   = get_data(lcfile)
        cont   = DRW_Model()
        cont.load_chain("chain_con.dat")
#        print(cont.hpd)
        rmap   = Rmap_Model(zylc)
#        p_ini = [np.log(2.0), np.log(100.0), 50, 3, 2]
#        rmap.do_map(p_ini, fixed=None, conthpd=cont.hpd, set_verbose=True)

        lags_ini = [0.0, 50.0]
        rmap.do_mcmc(cont.hpd, lags_ini=lags_ini, nwalkers=100, nburn=100,
                nchain=100, fburn="burntest.dat", fchain="chaintest.dat")

#        rmap.load_chain("chain50.dat")
#        rmap.get_hpd()
#        rmap.show_hist()

