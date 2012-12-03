#Last-modified: 03 Dec 2012 05:45:22 PM

from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
#np.seterr(all='raise')
from numpy.random import normal, multivariate_normal
from scipy.optimize import fmin
import matplotlib.pyplot as plt

from zylc import LightCurve, get_data
from cov import get_covfunc_dict
from spear import spear, spear_threading
from predict import PredictSignal, PredictRmap
from gp import FullRankCovariance, NearlyFullRankCovariance
from err import *
from emcee import EnsembleSampler
from itertools import groupby
from graphic import figure_handler

my_neg_inf = float(-1.0e+300)
my_pos_inf = float( 1.0e+300)

tau_floor     = 1.e-6
tau_ceiling   = 1.e+5
sigma_floor   = 1.e-6
sigma_ceiling = 1.e+2
logtau_floor     = np.log(tau_floor) 
logtau_ceiling   = np.log(tau_ceiling)   
logsigma_floor   = np.log(sigma_floor)  
logsigma_ceiling = np.log(sigma_ceiling) 

nu_floor      = 1.e-6
lognu_floor   = np.log(nu_floor)   
nu_ceiling    = 1.e+3
lognu_ceiling = np.log(nu_ceiling)   


def unpacksinglepar(p, covfunc="drw", uselognu=False) :
    """ Unpack the physical parameters from input 1-d array for single mode.
    """
    if p[0] > logsigma_ceiling :
        sigma = sigma_ceiling
    elif p[0] < logsigma_floor :
        sigma = sigma_floor
    else :
        sigma = np.exp(p[0])
    if p[1] > logtau_ceiling :
        tau = tau_ceiling
    elif p[1] < logtau_floor :
        tau = tau_floor
    else :
        tau = np.exp(p[1])

    if covfunc == "drw" :
        nu = None
    elif uselognu :
        if p[2] < lognu_floor :
            nu = nu_floor 
        elif p[2] > lognu_ceiling :
            nu = nu_ceiling 
        else :
            nu = np.exp(p[2])
    else :
        nu = p[2]
    return(sigma, tau, nu)


def unpackspearpar(p, nlc=None, hascontlag=False) :
    """ Unpack the physical parameters from input 1-d array for spear mode.
    """
    if nlc is None:
        # possible to figure out nlc from the size of p
        nlc = (len(p) - 2)//3 + 1
    sigma   = np.exp(p[0])
    tau     = np.exp(p[1])
    if hascontlag :
        lags    = np.zeros(nlc)
        wids    = np.zeros(nlc)
        scales  =  np.ones(nlc)
        for i in xrange(1, nlc) : 
            lags[i]   = p[2+(i-1)*3]
            wids[i]   = p[3+(i-1)*3]
            scales[i] = p[4+(i-1)*3]
        return(sigma, tau, lags, wids, scales)
    else :
        llags   = np.zeros(nlc-1)
        lwids   = np.zeros(nlc-1)
        lscales =  np.ones(nlc-1)
        for i in xrange(nlc-1) : 
            llags[i]   = p[2+i*3]
            lwids[i]   = p[3+i*3]
            lscales[i] = p[4+i*3]
        return(sigma, tau, llags, lwids, lscales)

def lnpostfn_single_p(p, zydata, covfunc, uselognu=False, set_prior=True,
        conthpd=None, rank="Full",
        set_retq=False, set_verbose=False) :
    """
    """
    sigma, tau, nu = unpacksinglepar(p, covfunc, uselognu=uselognu)
    if set_retq :
        vals = list(lnlikefn_single(zydata, covfunc=covfunc, rank=rank,
                    sigma=sigma, tau=tau, nu=nu,
                    set_retq=True, set_verbose=set_verbose))
    else :
        logl = lnlikefn_single(zydata, covfunc=covfunc, rank=rank,
                    sigma=sigma, tau=tau, nu=nu,
                    set_retq=False, set_verbose=set_verbose)
    prior = 0.0
    if set_prior :
        if covfunc == "kepler2_exp" :
            if conthpd is None :
                raise RuntimeError("kepler2_exp prior requires conthpd")
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
            # final
            prior += -0.5*(prior0*prior0+prior1*prior1)
        else :
            prior += - np.log(sigma)
            if tau > zydata.cont_cad :
                prior += - np.log(tau/zydata.cont_cad)
            elif tau < 0.001 :
                # 86.4 seconds if input is in days
                prior += my_neg_inf
            else :
                prior += - np.log(zydata.cont_cad/tau)

    if set_retq :
        vals[0] = vals[0] + prior
        vals.append(prior)
        return(vals)
    else :
        logp = logl + prior
        return(logp)

def lnlikefn_single(zydata, covfunc="drw", rank="Full", set_retq=False, 
        set_verbose=False, **covparams) :
    """
    """
    covfunc_dict = get_covfunc_dict(covfunc, **covparams)
    sigma = covparams.pop("sigma")
    tau   = covparams.pop("tau")
    nu    = covparams.pop("nu", None)
    # set up covariance function
    if (sigma<=0.0 or tau<=0.0) :
       return(_exit_with_retval(zydata.nlc, set_retq, 
              errmsg="Warning: illegal input of parameters", 
              set_verbose=set_verbose))
    if covfunc == "pow_exp" :
        if nu <= 0.0 or nu >= 2.0 :
            return(_exit_with_retval(zydata.nlc, set_retq, 
                   errmsg="Warning: illegal input of parameters in nu", 
                   set_verbose=set_verbose))
    elif covfunc == "matern" :
        if nu <= 0.0 :
            return(_exit_with_retval(zydata.nlc, set_retq, 
                   errmsg="Warning: illegal input of parameters in nu", 
                   set_verbose=set_verbose))
    elif covfunc == "kepler_exp" :
        # here nu is the ratio
        if nu < 0.0 or nu >= 1.0 :
            return(_exit_with_retval(zydata.nlc, set_retq, 
                   errmsg="Warning: illegal input of parameters in nu", 
                   set_verbose=set_verbose))
    elif covfunc == "kepler2_exp" :
        # here nu is the cutoff time scale
        if nu < 0.0 or nu >= tau :
            return(_exit_with_retval(zydata.nlc, set_retq, 
                   errmsg="Warning: illegal input of parameters in nu", 
                   set_verbose=set_verbose))

    # choice of ranks
    if rank == "Full" :
        # using full-rank
        C = FullRankCovariance(**covfunc_dict)
    elif rank == "NearlyFull" :
        # using nearly full-rank
        C = NearlyFullRankCovariance(**covfunc_dict)
    else :
        raise InputError("No such option for rank "+rank)
    # cholesky decompose S+N so that U^T U = S+N = C
    # using intrinsic method of C without explicitly writing out cmatrix
    try :
        U = C.cholesky(zydata.jarr, observed=False, nugget=zydata.varr)
    except :
        return(_exit_with_retval(zydata.nlc, set_retq,
               errmsg="Warning: non positive-definite covariance C",
               set_verbose=set_verbose))
    # calculate RPH likelihood
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq, set_verbose=set_verbose)
    return(retval)


def lnpostfn_spear_p(p, zydata, conthpd=None, lagtobaseline=0.3, laglimit=None,
        set_threading=False, blocksize=10000,
        set_retq=False, set_verbose=False):
    """ log-posterior function of p.

        Parameters
        ----------
        p : array_like
            Rmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1,
            ...]

        zydata: LightCurve object
            Light curve data.

        conthpd: ndarray, optional
            Priors on sigma and tau as an ndarray with shape (3, 2), 
            np.array([[log(sigma_low), log(tau_low)],
                      [log(sigma_med), log(tau_med)],
                      [log(sigma_hig), log(tau_hig)]])
            where 'low', 'med', and 'hig' are defined as the 68% confidence
            limits around the median. conthpd usually comes in as an attribute
            of the DRW_Model object DRW_Model.hpd (default: None).

        lagtobaseline: float, optional
            Prior on lags. When input lag exceeds lagtobaseline*baseline, a
            logarithmic prior will be applied.

        set_threading: bool, optional
            True if you want threading in filling matrix. It conflicts with the
            'threads' option in Rmap_Model.run_mcmc (default: False).

        blocksize: int, optional
            Maximum matrix block size in threading (default: 10000).

        set_retq: bool, optional
            Return the value(s) of q along with each component of the
            log-likelihood if True (default: False).

        set_verbose: bool, optional
            True if you want verbosity (default: False).

    """
    # unpack the parameters from p
    sigma, tau, llags, lwids, lscales = unpackspearpar(p, zydata.nlc,
            hascontlag=False)
    if set_retq :
        vals = list(lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
                set_retq=True, set_verbose=set_verbose,
                set_threading=set_threading, blocksize=blocksize))
    else :
        logl = lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
                set_retq=False, set_verbose=set_verbose,
                set_threading=set_threading, blocksize=blocksize)
    # conthpd is in natural log
    if conthpd is not None : 
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
    else :
        prior0 = 0.0
        prior1 = 0.0
    # for each lag
    prior2 = 0.0
    for i in xrange(zydata.nlc-1) :
        if lagtobaseline < 1.0 :
            if np.abs(llags[i]) > lagtobaseline*zydata.rj :
                # penalize long lags when they are larger than 0.3 times the baseline,
                # as it is too easy to fit the model with non-overlapping
                # signals in the light curves.
                prior2 += np.log(np.abs(llags[i])/(lagtobaseline*zydata.rj))
        # penalize long lags to be impossible
        if laglimit is not None :
            if llags[i] > laglimit[i][1] or llags[i] < laglimit[i][0] :
                # try not stack priors
                prior2 = my_pos_inf
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1) - prior2
    if set_retq :
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else :
        logp = logl + prior
        return(logp)

def lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
        set_retq=False, set_verbose=False, 
        set_threading=False, blocksize=10000):
    """ Log-likelihood function.
    """
    if zydata.issingle:
        raise UsageError("lnlikefn_spear does not work for single mode")
    # impossible scenarios
    if (sigma<=0.0 or tau<=0.0 or np.min(lwids)<0.0 or np.min(lscales)<=0.0
                   or np.max(np.abs(llags))>zydata.rj) :
       return(_exit_with_retval(zydata.nlc, set_retq, 
              errmsg="Warning: illegal input of parameters", 
              set_verbose=set_verbose))
    # fill in lags/wids/scales
    lags  = np.zeros(zydata.nlc)
    wids  = np.zeros(zydata.nlc)
    scales = np.ones(zydata.nlc)
    lags[1:]   = llags
    wids[1:]   = lwids
    scales[1:] = lscales
    # calculate covariance matrix
    if set_threading :
        C = spear_threading(zydata.jarr,zydata.jarr,
              zydata.iarr,zydata.iarr,sigma,tau,lags,wids,scales, 
              blocksize=blocksize)
    else :
        C = spear(zydata.jarr,zydata.jarr,
              zydata.iarr,zydata.iarr,sigma,tau,lags,wids,scales)
    # decompose C inplace
    U, info = cholesky(C, nugget=zydata.varr, inplace=True, raiseinfo=False)
    # handle exceptions here
    if info > 0 :
       return(_exit_with_retval(zydata.nlc, set_retq, 
              errmsg="Warning: non positive-definite covariance C", 
              set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq, set_verbose=set_verbose)
    return(retval)


def _lnlike_from_U(U, zydata, set_retq=False, set_verbose=False):
    """ calculate the log-likelihoods from the upper triangle of cholesky
    decomposition.
    """
    # log determinant of C^-1
    detC_log = chodet_from_tri(U, retlog=True)
    # solve for C a = y so that a = C^-1 y 
    a = chosolve_from_tri(U, zydata.marr)
    # solve for C b = L so that b = C^-1 L
    b = chosolve_from_tri(U, zydata.larr)
    # multiply L^T and b so that C_p = L^T C^-1 L = C_q^-1
    C_p = np.dot(zydata.larrTr, b)
    # for 'issingle is True' case, C_p is a scalar.
    if np.isscalar(C_p):
        # for single-mode, cholesky of C_p is simply squre-root of C_p
        W = np.sqrt(C_p)
        detCp_log = np.log(C_p.squeeze())
        # for single-mode, simply devide L^T by C_p
        d = zydata.larrTr/C_p
    else:
        # cholesky decompose C_p so that W^T W = C_p
        W, info = cholesky(C_p, raiseinfo=False)
        if info > 0 :
            return(_exit_with_retval(zydata.nlc, set_retq, 
                errmsg="Warning: non positive-definite covariance W", 
                set_verbose=set_verbose))
        detCp_log = chodet_from_tri(W, retlog=True)
        # solve for C_p d = L^T so that d = C_p^-1 L^T = C_q L^T
        d = chosolve_from_tri(W, zydata.larrTr)
    # multiply b d and a so that e = C^-1 L C_p^-1 L^T C^-1 y 
    e = np.dot(b, np.dot(d, a))
    # a minus e so that f = a - e = C^-1 y - C^-1 L C_p^-1 L^T C^-1 y
    #              thus f = C_v^-1 y
    f = a - e
    # multiply y^T  and f so that h = y^T C_v^-1 y
    h = np.dot(zydata.marr, f)
    # chi2_PRH = -0.5*h
    _chi2 = -0.5*h
    # following Carl Rasmussen's term, a penalty on the complexity of 
    # the model
    _compl_pen = -0.5*detC_log
    # penalty on blatant linear drift
    _wmean_pen = -0.5*detCp_log
    # final log_likelhood
    _log_like = _chi2 + _compl_pen + _wmean_pen
    if set_retq:
        q = np.dot(d, a)
        return(_log_like, _chi2, _compl_pen, _wmean_pen, q)
    else:
        return(_log_like)

def _exit_with_retval(nlc, set_retq, errmsg=None, set_verbose=False):
    """ Return failure elegantly.

    When you are desperate and just want to leave the calculation with 
    appropriate return values that quietly speak out your angst.
    """
    if errmsg is not None:
        if set_verbose:
            print("Exit: %s"%errmsg)
    if set_retq :
        return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf, 
              [my_neg_inf]*nlc)
    else:
        return(my_neg_inf)


class Cont_Model(object) :
    def __init__(self, zydata=None, covfunc="drw") :
        """ Cont Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        """
        self.zydata  = zydata
        self.covfunc = covfunc
        if zydata is None :
            pass
        else :
            self.nlc      = zydata.nlc
            self.npt      = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_cad_min = zydata.cont_cad_min
            self.cont_cad_max = zydata.cont_cad_max
            self.cont_std = zydata.cont_std
            self.rj       = zydata.rj
            self.jstart   = zydata.jstart
            self.jend     = zydata.jend
            self.names    = zydata.names
        self.vars = ["sigma", "tau"]
        self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
        if covfunc == "drw" :
            self.uselognu = False
            self.ndim = 2
        elif covfunc == "matern" or covfunc == "kepler2_exp" :
            self.uselognu = True
            self.ndim = 3
            self.vars.append("nu")
            self.texs.append(r"$\log\,\nu$")
        else :
            self.uselognu = False
            self.ndim = 3
            self.vars.append("nu")
            self.texs.append(r"$\nu$")

    def __call__(self, p, set_prior=True, rank="Full", set_retq=False,
            set_verbose=True): 
        return(lnpostfn_single_p(p, self.zydata, self.covfunc, 
            uselognu=self.uselognu, set_prior=set_prior, rank=rank,
            set_retq=set_retq, set_verbose=set_verbose))

    def do_map(self, p_ini, fixed=None, **lnpostparams) :
        """
        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        set_retq    = lnpostparams.pop("set_retq",    False)
        set_prior   = lnpostparams.pop("set_prior",   True)
        rank        = lnpostparams.pop("rank",       "Full")
        if set_retq is True :
            raise InputError("set_retq has to be False")
        p_ini = np.asarray(p_ini)
        if fixed is not None :
            fixed = np.asarray(fixed)
            func = lambda _p : -lnpostfn_single_p(_p*fixed+p_ini*(1.-fixed),
                    self.zydata, self.covfunc, uselognu=self.uselognu, 
                    set_retq=False,
                    set_prior=set_prior, rank=rank, set_verbose=set_verbose)
        else :
            func = lambda _p : -lnpostfn_single_p(_p,
                    self.zydata, self.covfunc, uselognu=self.uselognu, 
                    set_retq=False,
                    set_prior=set_prior, rank=rank, set_verbose=set_verbose)
        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        sigma, tau, nu = unpacksinglepar(p_bst, covfunc=self.covfunc,
                uselognu=self.uselognu)
        if fixed is not None :
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        if set_verbose :
            print("Best-fit parameters are:")
            print("sigma %8.3f tau %8.3f"%(sigma, tau))
            if nu is not None :
                print("nu %8.3f"%nu)
            print("with logp  %10.5g "%-v_bst)
        return(p_bst, -v_bst)

    def do_grid1d(self, p_ini, fixed, rangex, dx, fgrid1d, **lnpostparams) :
        set_verbose = lnpostparams.pop("set_verbose", True)
        xs = np.arange(rangex[0], rangex[-1]+dx, dx)
        fixed = np.asarray(fixed)
        nfixed = np.sum(fixed == 0)
        if nfixed != 1 :
            raise InputError("wrong number of fixed pars ")
        f = open(fgrid1d, "w")
        for x in xs :
            _p_ini = p_ini*fixed + x*(1.-fixed) 
            _p, _l = self.do_map(_p_ini, fixed=fixed, **lnpostparams)
            _line = "".join([format(_l, "20.10g"), 
                             " ".join([format(r, "10.5f") for r in _p]), "\n"])
            f.write(_line)
            f.flush()
        f.close()
        if set_verbose :
            print("saved grid1d result to %s"%fgrid1d)

    def do_grid2d(self, p_ini, fixed, rangex, dx, rangey, dy, fgrid2d, 
            **lnpostparams) :
        fixed = np.asarray(fixed)
        set_verbose = lnpostparams.pop("set_verbose", True)
        xs = np.arange(rangex[0], rangex[-1]+dx, dx)
        ys = np.arange(rangey[0], rangey[-1]+dy, dy)
        nfixed = np.sum(fixed == 0)
        if nfixed != 2 :
            raise InputError("wrong number of fixed pars ")
        posx, posy = np.nonzero(1-fixed)[0]
        dimx, dimy = len(xs),len(ys)
        header = " ".join(["#", str(posx), str(posy), str(dimx), str(dimy), "\n"])
        print(header)
        f = open(fgrid2d, "w")
        f.write(header)
        for x in xs :
            for y in ys :
                _p_ini = p_ini*fixed
                _p_ini[posx] = x
                _p_ini[posy] = y
                _p, _l = self.do_map(_p_ini, fixed=fixed, **lnpostparams)
                _line = "".join([format(_l, "20.10g"), 
                             " ".join([format(r, "10.5f") for r in _p]), "\n"])
                f.write(_line)
                f.flush()
        f.close()
        if set_verbose :
            print("saved grid2d result to %s"%fgrid2d)

    def read_logp_map(self, fgrid2d, set_verbose=True) :
        f = open(fgrid2d, "r")
        posx, posy, dimx, dimy = [int(r) for r in f.readline().lstrip("#").split()]
        if set_verbose :
            print("grid file %s is registered for"%fgrid2d)
            print("var_x = %10s var_y = %10s"%(self.vars[posx], self.vars[posy]))
            print("dim_x = %10d dim_y = %10d"%(dimx, dimy))
        if self.covfunc != "drw" :
            logp, sigma, tau, nu = np.genfromtxt(f,unpack=True,usecols=(0,1,2,3))
        else :
            logp, sigma, tau     = np.genfromtxt(f,unpack=True,usecols=(0,1,2))
        f.close()
        retdict = {
                   'logp'   :    logp.reshape(dimx, dimy).T,
                   'sigma'  :   sigma.reshape(dimx, dimy).T,
                   'tau'    :     tau.reshape(dimx, dimy).T,
                   'nu'     :     None,
                   'posx'   :   posx,
                   'posy'   :   posy,
                   'dimx'   :   dimx,
                   'dimy'   :   dimy,
                  }
        if self.covfunc != "drw" :
            retdict['nu'] = nu.reshape(dimx, dimy).T
        return(retdict)

    def show_logp_map(self, fgrid2d, set_normalize=True, vmin=None, vmax=None,
            set_contour=True, clevels=None,
            set_verbose=True, figout=None, figext=None) :
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(8,8))
        ax  = fig.add_subplot(111)
        retdict = self.read_logp_map(fgrid2d, set_verbose=set_verbose)
        x = retdict[self.vars[retdict['posx']]]/ln10
        y = retdict[self.vars[retdict['posy']]]/ln10
        z = retdict['logp']
        if x is None or y is None :
            raise InputError("incompatible fgrid2d file"+fgrid2d)
        xmin,xmax,ymin,ymax = np.min(x),np.max(x),np.min(y),np.max(y)
        extent = (xmin,xmax,ymin,ymax)
        if set_normalize :
            zmax = np.max(z)
            z    = z - zmax
        if vmin is None:
            vmin = z.min()
        if vmax is None:
            vmax = z.max()
        im = ax.imshow(z, origin='lower', vmin=vmin, vmax=vmax,
                          cmap='jet', interpolation="nearest", aspect="auto", 
                          extent=extent)
        if set_contour:
            if clevels is None:
                sigma3,sigma2,sigma1 = 11.8/2.0,6.17/2.0,2.30/2.0
                levels = (vmax-sigma1, vmax-sigma2, vmax-sigma3)
            else:
                levels = clevels
            ax.set_autoscale_on(False)
            cs = ax.contour(z,levels, hold='on',colors='k',
                              origin='lower',extent=extent)
        ax.set_xlabel(self.texs[retdict['posx']])
        ax.set_ylabel(self.texs[retdict['posy']])
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def do_mcmc(self, conthpd=None, set_prior=True, rank="Full", 
            nwalkers=100, nburn=50, nchain=50,
            fburn=None, fchain=None, flogp=None, threads=1, set_verbose=True):
        """
        """
        # initialize a multi-dim random number array 
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initial values of sigma to be scattering around cont_std
        p0[:,0] = p0[:,0] - 0.5 + np.log(self.cont_std)
        # initial values of tau   filling 0 - 0.5rj
        p0[:,1] = np.log(self.rj*0.5*p0[:,1])
        if self.covfunc == "pow_exp" :
            p0[:,2] = p0[:,2] * 1.99
        elif self.covfunc == "matern" :
            p0[:,2] = np.log(p0[:,2] * 5)
        elif self.covfunc == "kepler2_exp" :
            p0[:,2] = np.log(self.rj*0.1*p0[:,2])

        if set_verbose :
            print("start burn-in")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"%
                    (nburn, nwalkers, nburn*nwalkers))
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_single_p, 
                    args=(self.zydata, self.covfunc, self.uselognu, 
                        set_prior, conthpd, rank, False, False), 
                    threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose :
            print("burn-in finished")
        if fburn is not None :
            if set_verbose :
                print("save burn-in chains to %s"%fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset sampler
        sampler.reset()
        if set_verbose :
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose :
            print("sampling finished")
        af = sampler.acceptance_fraction
        if set_verbose :
            print("acceptance fractions for all walkers are")
            print(" ".join([format(r, "3.2f") for r in af]))
        if fchain is not None :
            if set_verbose :
                print("save MCMC chains to %s"%fchain)
            np.savetxt(fchain, sampler.flatchain)
        if flogp is not None :
            if set_verbose :
                print("save logp of MCMC chains to %s"%flogp)
            np.savetxt(flogp, np.ravel(sampler.lnprobability), fmt='%16.8f')
        # make chain an attritue
        self.flatchain = sampler.flatchain
        self.flatchain_whole = np.copy(self.flatchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

    def get_hpd(self, set_verbose=True):
        """
        """
        hpd = np.zeros((3, self.ndim))
        chain_len = self.flatchain.shape[0]
        pct1sig = chain_len*np.array([0.16, 0.50, 0.84])
        medlowhig = pct1sig.astype(np.int32)
        for i in xrange(self.ndim):
            vsort = np.sort(self.flatchain[:,i])
            hpd[:,i] = vsort[medlowhig]
            if set_verbose :
                print("HPD of %s"%self.vars[i])
                if (self.vars[i] == "nu" and (not self.uselognu)) :
                    print("low: %8.3f med %8.3f hig %8.3f"%tuple(hpd[:,i]))
                else :
                    print("low: %8.3f med %8.3f hig %8.3f"%tuple(np.exp(hpd[:,i])))
        # register hpd to attr
        self.hpd = hpd

    def show_hist(self, bins=100, figout=None, figext=None):
        """
        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig  = plt.figure(figsize=(8, 5))
        for i in xrange(self.ndim) :
            ax = fig.add_subplot(1,self.ndim,i+1)
            if (self.vars[i] == "nu" and (not self.uselognu)) :
                ax.hist(self.flatchain[:,i], bins)
                if self.covfunc == "kepler2_exp" :
                    ax.axvspan(self.cont_cad_min,
                            self.cont_cad, color="g", alpha=0.2)
            else :
                ax.hist(self.flatchain[:,i]/ln10, bins)
                if self.vars[i] == "nu" and self.covfunc == "kepler2_exp" :
                    ax.axvspan(np.log10(self.cont_cad_min),
                               np.log10(self.cont_cad), color="g", alpha=0.2)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        #plt.get_current_fig_manager().toolbar.zoom()
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def load_chain(self, fchain, set_verbose=True):
        """
        """
        if set_verbose :
            print("load MCMC chain from %s"%fchain)
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

    def break_chain(self, covpar_segments):
        """ Break the chain.
        """
        if (len(covpar_segments) != self.ndim) :
            print("Error: covpar_segments has to be a list of length %d"%(self.ndim))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        for i, covpar_seq in enumerate(covpar_segments) : 
            if covpar_seq is None:
                continue
            indx = np.argsort(self.flatchain[:, i])
            imin, imax = np.searchsorted(self.flatchain[indx, i], covpar_seq)
            indx_cut = indx[imin : imax]
            if len(indx_cut) < 10 :
                print("Warning: cut too aggressive!")
                return(1)
            self.flatchain = self.flatchain[indx_cut, :]

    def restore_chain(self) :
        self.flatchain = np.copy(self.flatchain_whole)

    def get_qlist(self, p_bst) :
        """ get the linear responses.
        """
        self.qlist = lnpostfn_single_p(p_bst, self.zydata, self.covfunc,
                uselognu=self.uselognu, rank="Full", set_retq=True)[4]

    def do_pred(self, p_bst, fpred=None, dense=10, rank="Full",
            set_overwrite=True) :
        """
        """
        self.get_qlist(p_bst)
        self.zydata.update_qlist(self.qlist)
        sigma, tau, nu = unpacksinglepar(p_bst, self.covfunc, uselognu=self.uselognu)
        lcmean=self.zydata.blist[0]
        P = PredictSignal(zydata=self.zydata, lcmean=lcmean,
                rank=rank, covfunc=self.covfunc,
                sigma=sigma, tau=tau, nu=nu)
        nwant = dense*self.cont_npt
        jwant0 = self.jstart - 0.1*self.rj
        jwant1 = self.jend   + 0.1*self.rj
        jwant = np.linspace(jwant0, jwant1, nwant)
        mve, var = P.mve_var(jwant)
        sig = np.sqrt(var)
        zylclist_pred = [[jwant, mve, sig],]
        zydata_pred   = LightCurve(zylclist_pred)
        if fpred is not None :
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        return(zydata_pred)


class Rmap_Model(object) :
    def __init__(self, zydata=None) :
        """ Rmap Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        """
        self.zydata = zydata
        if zydata is None :
            pass
        else :
            self.nlc      = zydata.nlc
            self.npt      = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj       = zydata.rj
            self.jstart   = zydata.jstart
            self.jend     = zydata.jend
            self.names    = zydata.names
            # number of parameters
            self.ndim  = 2 + (self.nlc-1)*3
            self.vars  = [ "sigma",           "tau"         ]
            self.texs  = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            for i in xrange(1, self.nlc) :
                self.vars.append("_".join(["lag",   self.names[i]]))
                self.vars.append("_".join(["wid",   self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append( "".join([r"$t_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$w_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$s_{", self.names[i] ,r"}$"]))

    def __call__(self, p, conthpd=None, lagtobaseline=0.3, set_retq=False,
            set_verbose=True, set_threading=False, blocksize=10000) :
        return(lnpostfn_spear_p(p, self.zydata, conthpd=conthpd, 
            lagtobaseline=lagtobaseline,
            set_retq=set_retq, set_verbose=set_verbose,
            set_threading=set_threading, blocksize=blocksize,
            ))

    def do_map(self, p_ini, fixed=None, **lnpostparams) :
        """ Do an optimization to find the Maximum a Posterior estimates.

        Parameters
        ----------
        p_ini: array_like
            DRW_Model parameters [log(sigma), log(tau)].

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1] means sigma is fixed while tau is varying. fixed=[1,
            1] is equivalent to fixed=None (default:
            None).

        Returns
        -------
        p_bst : array_like
            Best-fit parameters.

        l: float
            The maximum log-posterior.

        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        set_retq    = lnpostparams.pop("set_retq",    False)
        if set_retq is True :
            raise InputError("set_retq has to be False")
        p_ini = np.asarray(p_ini)
        if fixed is not None :
            fixed = np.asarray(fixed)
            func = lambda _p : -lnpostfn_spear_p(_p*fixed+p_ini*(1.-fixed),
                    self.zydata, **lnpostparams)
        else :
            func = lambda _p : -lnpostfn_spear_p(_p, 
                    self.zydata, **lnpostparams)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None :
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, llags, lwids, lscales = unpackspearpar(p_bst,
                self.zydata.nlc, hascontlag=False)
        if set_verbose :
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f"%(sigma, tau))
            for i in xrange(self.nlc-1) :
                ip = 2+i*3
                print("%s %8.3f %s %8.3f %s %8.3f"%(
                    self.vars[ip+0], llags[i],
                    self.vars[ip+1], lwids[i],
                    self.vars[ip+2], lscales[i],
                    ))
            print("with logp  %10.5g "%-v_bst)
        return(p_bst, -v_bst)


    def do_mcmc(self, conthpd=None, lagtobaseline=0.3, laglimit="baseline",
            nwalkers=100, nburn=100, nchain=100, threads=1, 
            fburn=None, fchain=None, flogp=None,
            set_threading=False, blocksize=10000,
            set_verbose=True):
        """ test
        """
        if (threads > 1 and (not set_threading)):
            if set_verbose:
                print("run parallel chains of number %2d "%threads)
        elif (threads == 1) :
            if set_verbose:
                if set_threading :
                    print("run single chain in submatrix blocksize %10d "%blocksize)
                else :
                    print("run single chain without subdividing matrix ")
        else :
            raise InputError("conflicting set_threading and threads setup")
        if laglimit == "baseline" :
            laglimit = [[-self.rj, self.rj],]*(self.nlc-1)
        elif len(laglimit) != (self.nlc - 1) :
            raise InputError("laglimit should be a list of lists matching number of lines")
        # generate array of random numbers
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initialize array
        if conthpd is None:
            p0[:, 0] += np.log(self.cont_std)-0.5
            p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        else :
            p0[:, 0] += conthpd[1,0]-0.5
            p0[:, 1] += conthpd[1,1]-0.5
        for i in xrange(self.nlc-1) :
#            p0[:, 2+i*3] = p0[:,2+i*3]*self.rj*lagtobaseline
            p0[:, 2+i*3] = p0[:,2+i*3]*(laglimit[i][1]-laglimit[i][0]) + laglimit[i][0]
        if set_verbose :
            print("start burn-in")
            if conthpd is None :
                print("no priors on sigma and tau")
            else :
                print("using priors on sigma and tau from the continuum fitting")
                print(np.exp(conthpd))
            if lagtobaseline < 1.0 :
                print("penalize lags longer than %3.2f of the baseline"%lagtobaseline)
            else :
                print("no penalizing long lags, but only restrict to within the baseline")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"%
                (nburn, nwalkers, nburn*nwalkers))
        # initialize the ensemble sampler
        sampler = EnsembleSampler(nwalkers, self.ndim,
                    lnpostfn_spear_p,
                    args=(self.zydata, conthpd, lagtobaseline, laglimit,
                          set_threading, blocksize, False, False), 
                    threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose :
            print("burn-in finished")
        if fburn is not None :
            if set_verbose :
                print("save burn-in chains to %s"%fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler
        sampler.reset()
        if set_verbose :
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose :
            print("sampling finished")
        af = sampler.acceptance_fraction
        if set_verbose :
            print("acceptance fractions are")
            print(" ".join([format(r, "3.2f") for r in af]))
        if fchain is not None :
            if set_verbose :
                print("save MCMC chains to %s"%fchain)
            np.savetxt(fchain, sampler.flatchain)
        if flogp is not None :
            if set_verbose :
                print("save logp of MCMC chains to %s"%flogp)
            np.savetxt(flogp, np.ravel(sampler.lnprobability), fmt='%16.8f')
        # make chain an attritue
        self.flatchain = sampler.flatchain
        self.flatchain_whole = np.copy(self.flatchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

    def get_hpd(self, set_verbose=True):
        """ Get the 68% percentile range of each parameter to self.hpd.

        Parameters
        ----------
        set_verbose: bool, optional
            True if you want verbosity (default: True).

        """
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

    def show_hist(self, bins=100, lagbinsize=1.0, figout=None, figext=None):
        """ Plot the histograms.
        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig  = plt.figure(figsize=(14, 2.8*self.nlc))
        for i in xrange(2) :
            ax = fig.add_subplot(self.nlc,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        for k in xrange(self.nlc-1):
            for i in xrange(2+k*3, 5+k*3) :
                ax = fig.add_subplot(self.nlc,3,i+1+1) 
                if np.mod(i, 3) == 2 : 
                    # lag plots
                    lagbins = np.arange(int(np.min(self.flatchain[:,i])), 
                            int(np.max(self.flatchain[:,i]))+lagbinsize, lagbinsize)
                    ax.hist(self.flatchain[:,i], bins=lagbins)
                else :
                    ax.hist(self.flatchain[:,i], bins)
                ax.set_xlabel(self.texs[i])
                ax.set_ylabel("N")
#        plt.get_current_fig_manager().toolbar.zoom()
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def break_chain(self, llag_segments):
        """ Break the chain.

        Parameters
        ----------
        llag_segments: list of lists
            list of length self.nlc-1, wich each element a two-element array
            bracketing the range of lags (usually the single most probable peak) 
            you want to consider for each line.
        
        """
        if (len(llag_segments) != self.nlc-1) :
            print("Error: llag_segments has to be a list of length %d"%(self.nlc-1))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        for i, llag_seq in enumerate(llag_segments) : 
            if llag_seq is None:
                continue
            indx = np.argsort(self.flatchain[:, 2+i*3])
            imin, imax = np.searchsorted(self.flatchain[indx, 2+i*3], llag_seq)
            indx_cut = indx[imin : imax]
            self.flatchain = self.flatchain[indx_cut, :]

    def restore_chain(self) :
        self.flatchain = np.copy(self.flatchain_whole)

    def load_chain(self, fchain, set_verbose=True):
        """ Load stored MCMC chain.

        Parameters
        ----------
        fchain: string
            Name for the chain file.

        set_verbose: bool, optional
            True if you want verbosity (default: True).
        """
        if set_verbose :
            print("load MCMC chain from %s"%fchain)
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)


    def get_qlist(self, p_bst):
        self.qlist = lnpostfn_spear_p(p_bst, self.zydata, conthpd=None, lagtobaseline=0.3, 
                    set_threading=True, blocksize=10000,
                    set_retq=True, set_verbose=False)[4]

    def do_pred(self, p_bst, fpred=None, dense=10, set_overwrite=True) :
        """ Calculate the predicted mean and variance of each light curve on a
        densely sampled time axis.

        Parameters
        ----------
        p_bst: array_like
            Input paraemeters.

        fpred: string, optional
            Name of the output file for the predicted light curves, set it to
            None if you do not want output (default: None).

        dense: int, optional
            The factor by which the predicted light curves should be more
            densely sampled than the original data (default: 10).

        set_overwrite: bool, optional
            True if you want to overwrite existing fpred (default: True).

        Returns
        -------
        zydata_pred: LightCurve object
            Predicted light curves packaged as a LightCurve object.

        """
        self.get_qlist(p_bst)
        sigma, tau, lags, wids, scales = unpackspearpar(p_bst,
                self.zydata.nlc, hascontlag=True)
        # update qlist
        self.zydata.update_qlist(self.qlist)
        # initialize PredictRmap object
        P = PredictRmap(zydata=self.zydata, sigma=sigma, tau=tau, 
                lags=lags, wids=wids, scales=scales)
        nwant = dense*self.cont_npt
        jwant0 = self.jstart - 0.1*self.rj
        jwant1 = self.jend   + 0.1*self.rj
        jwant = np.linspace(jwant0, jwant1, nwant)
        zylclist_pred = []
        for i in xrange(self.nlc) :
            iwant = np.ones(nwant)*(i+1)
            mve, var = P.mve_var(jwant, iwant)
            sig = np.sqrt(var)
            zylclist_pred.append([jwant, mve, sig])
        zydata_pred   = LightCurve(zylclist_pred)
        if fpred is not None :
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        return(zydata_pred)



if __name__ == "__main__":    
    import matplotlib.pyplot as plt

    if False :
        lcfile = "dat/loopdeloop_con.dat"
        zydata   = get_data(lcfile)
        cont   = Cont_Model(zydata, "drw")
        cont.do_mcmc(set_prior=True, rank="Full",
                nwalkers=100, nburn=50, nchain=50, fburn=None,
                fchain="chain0.dat", threads=2)
#        cont.load_chain("chain0.dat")
#        cont.show_hist()
#        p_bst = [cont.hpd[1, 0], cont.hpd[1,1]]
#        p_bst = cont.do_map(p_bst, fixed=None, set_prior=False, rank="Full", 
#            set_verbose=True)[0]
#        zypred = cont.do_pred(p_bst, fpred="dat/loopdeloop_con.p.dat", dense=10)
#        zypred.plot(set_pred=True, obs=zydata)

    if False :
        lcfile = "dat/loopdeloop_con_y.dat"
        zydata   = get_data(lcfile)
        rmap   = Rmap_Model(zydata)

        cont   = DRW_Model()
        cont.load_chain("dat/chain0.dat")

        rmap.do_mcmc(conthpd=cont.hpd, nwalkers=100, nburn=1,
                nchain=5, fburn=None, fchain="dat/test.dat",
                flogp="dat/logp.dat", threads=2)

#        rmap.load_chain("dat/test.dat")
#        rmap.show_hist()
#        rmap.break_chain([[10, 200],])
#        rmap.show_hist()

#        p_ini = [np.log(3.043), np.log(170.8), 232.8, 0.868, 1.177]
#        rmap.do_map(p_ini, fixed=None, conthpd=cont.hpd, set_verbose=True)

#        p_bst = [np.log(3.043), np.log(170.8), 232.8, 0.868, 1.177]
#        zypred = rmap.do_pred(p_bst, fpred="dat/loopdeloop_con_y.p.dat", dense=10)
#        zypred.plot(set_pred=True, obs=zydata)

    if False :
        lcfile = "dat/loopdeloop_con_y_z.dat"
        zydata   = get_data(lcfile)
        rmap   = Rmap_Model(zydata)

        cont   = DRW_Model()
        cont.load_chain("dat/chain0.dat")
         
#        rmap.do_mcmc(conthpd=cont.hpd, nwalkers=100, nburn=50,
#                nchain=50, fburn=None, fchain="dat/test2.dat", threads=2)


        rmap.load_chain("dat/test2.dat")
#        rmap.show_hist()
        rmap.break_chain([[0,300],[0,400]])
        rmap.show_hist()
        rmap.get_hpd()
        p_bst = rmap.hpd[1,:]
        p_bst2 = rmap.do_map(p_bst, fixed=None, conthpd=cont.hpd, lagtobaseline=0.3,
                set_verbose=True)[0]
        zypred = rmap.do_pred(p_bst2, fpred="dat/test.p.dat", dense=10)
        zypred.plot(set_pred=True, obs=zydata)

    if False :
        lcfile = "dat/Arp151/Arp151_B.dat"
        bchain = "dat/Arp151/chain_arp151_B.dat"
        zydata   = get_data(lcfile)
        cont   = Cont_Model(zydata)
#        cont.do_mcmc(nwalkers=100, nburn=50, nchain=50, fburn=None,
#                fchain=bchain)
        cont.load_chain(bchain)
#        cont.show_hist()

    if False :

        bchain = "dat/Arp151/chain_arp151_B.dat"
        fcont  = "dat/Arp151/Arp151_B.dat"
        zydata = get_data(fcont)
        cont   = Cont_Model(zydata)
        cont.load_chain(bchain)

        lcfiles = [
                   "dat/Arp151/Arp151_V.dat",
#                   "dat/Arp151/Arp151_Halpha.dat",
#                   "dat/Arp151/Arp151_Hbeta.dat",
#                   "dat/Arp151/Arp151_Hgamma.dat",
#                   "dat/Arp151/Arp151_HeI.dat",
#                   "dat/Arp151/Arp151_HeII.dat",
                   ]
        for i, lcfile in enumerate(lcfiles) :
            fchain = "dat/Arp151/chain_arp151_" + str(i+1) + ".dat"
            zydata   = get_data([fcont, lcfile])
#            zydata.plot()
            rmap   = Rmap_Model(zydata)
#            rmap.do_mcmc(conthpd=cont.hpd, nwalkers=100, nburn=50,
#                nchain=50, fburn=None, fchain=fchain, threads=1)
            rmap.load_chain(fchain)
            rmap.show_hist(set_adaptive=True, bins=50, floor=100)
#            rmap.break_chain([[-20,80],])
#            rmap.show_hist()
#            rmap.get_hpd()



