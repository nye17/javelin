import numpy as np
import sys
# np.seterr(all='raise')
from scipy.optimize import fmin
import matplotlib.pyplot as plt
# internal packages
from javelin.cholesky_utils import (cholesky, chosolve_from_tri, 
                     chodet_from_tri)
from javelin.zylc import LightCurve
from javelin.cov import get_covfunc_dict
from javelin.spear import spear, spear_threading
from javelin.predict import (PredictSignal, PredictRmap, PredictPmap, 
                     PredictSPmap, PredictSCmap)
from javelin.gp import FullRankCovariance, NearlyFullRankCovariance
from javelin.err import InputError, UsageError
#from emcee import EnsembleSampler
from javelin.graphic import figure_handler
import pdb
import copy

from javelin.emcee_internal import EnsembleSampler

my_neg_inf = float(-1.0e+300)
my_pos_inf = float(+1.0e+300)

tau_floor = 100.        # Units of days
tau_ceiling = 300.
sigma_floor = 1.e-6
sigma_ceiling = 1.e+2
logtau_floor = np.log(tau_floor)
logtau_ceiling = np.log(tau_ceiling)
logsigma_floor = np.log(sigma_floor)
logsigma_ceiling = np.log(sigma_ceiling)

nu_floor = 1.e-6
lognu_floor = np.log(nu_floor)
nu_ceiling = 1.e+3
lognu_ceiling = np.log(nu_ceiling)


def _lnlike_from_U(U, zydata, set_retq=False, set_verbose=False):
    """ Calculate the log-likelihoods from the upper triangle of cholesky
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
        if info > 0:
            return(_exit_with_retval(
                zydata.nlc, set_retq,
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
            print("Exit: %s" % errmsg)
    if set_retq:
        return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf,
               [my_neg_inf]*nlc)
    else:
        return(my_neg_inf)


def _get_hpd(ndim, flatchain):
    """ Get the 68% percentile range of each parameter.
    """
    hpd = np.zeros((3, ndim))
    chain_len = flatchain.shape[0]
    pct1sig = chain_len*np.array([0.16, 0.50, 0.84])
    medlowhig = pct1sig.astype(np.int32)
    for i in xrange(ndim):
        vsort = np.sort(flatchain[:,i])
        hpd[:,i] = vsort[medlowhig]
    return(hpd)


def _get_bfp(flatchain, logp):
    j = np.argmax(logp)
    bfp = flatchain[j, :]
    return(bfp)

def lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, set_retq=False,
                   set_verbose=False, set_threading=False, blocksize=10000):
    """ Internal function to calculate the log likelihood.
    zydata = light curve data objects
    sigma = DRW amplitude
    tau = DRW timescale
    llags = lags for each light curve with respect to driving continuum
    lwids = widths for tophat transfer function applied to light curves
    lscales = scales for tophat transfer function applied to light curves
    """
    if zydata.issingle:
        raise UsageError("lnlikefn_spear does not work for single mode")
    # impossible scenarios
    if ((sigma <= 0.0 or tau <= 0.0 or np.min(lwids) < 0.0 or
         np.min(lscales) <= 0.0 or np.max(np.abs(llags)) > zydata.rj)):
        return(_exit_with_retval(zydata.nlc, set_retq,
                                 errmsg="Warning: illegal input of parameters",
                                 set_verbose=set_verbose))
    # fill in lags/wids/scales
    lags = np.zeros(zydata.nlc)
    wids = np.zeros(zydata.nlc)
    scales = np.ones(zydata.nlc)
    lags[1:] = llags # zero lag with respect to ref band, so start at element 1
    wids[1:] = lwids
    scales[1:] = lscales
    # calculate covariance matrix
    if set_threading:
        C = spear_threading(zydata.jarr, zydata.jarr, zydata.iarr, zydata.iarr,
                            sigma, tau, lags, wids, scales, blocksize=blocksize)
    else:
        C = spear(zydata.jarr, zydata.jarr, zydata.iarr, zydata.iarr, sigma,
                  tau, lags, wids, scales)
    # decompose C inplace
    U, info = cholesky(C, nugget=zydata.varr, inplace=True, raiseinfo=False)
    # handle exceptions here
    if info > 0:
        return(
            _exit_with_retval(
                zydata.nlc, set_retq,
                errmsg="Warning: non positive-definite covariance C",
                set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)





### If adding directly to lcmodel.py, start here
def thin_disk_func(a, b, waves, refwave):
    """ Thin-disk parameterization of the accretion disk. 
    a = the accretion disk size at the reference wavelength
    b = the power-law scaling of disk size as a function of wavelength (beta)
    waves = float or list/array of floats that will be the wavelengths at which
        the new disk sizes are to be calculated
    refwave = float corresponding to the wavelength upon which 'a' is 
        calculated; the wavelength of the driving light curve
    
    returns the disk size at each wavelength provided in 'waves', as a numpy 
        array
        
    """
    return a*(np.power((np.float(waves)/np.float(refwave)), np.float(b)) - 1.)

def unpackthindiskpar(p, nlc=None, hascontlag=False, lwaves=None, refwave=None):
    """ Internal Function: unpack the physical parameters from input 1-d
    array for thin disk mode.
    p = sigma, tau, a, b, width1, scale1, ..., widthn, scalen
    lwaves = array or list of floats that are wavelengths for the input light
        curves
    refwave = wavelength of the driving light curve
    
    returns necessary info from walkers to ready for lnlike
    """
    if nlc is None:
        # possible to figure out nlc from the size of p
        nlc = (len(p) - 4.)//2. + 1.
    if lwaves is None:
    	print "You need to provide values for \'lwaves\'."
        sys.exit()          
    sigma = np.exp(p[0])    # DRW amplitude
    tau = np.exp(p[1])      # DRW damping timescale
    alph = p[2]             # Thin disk normalization; disk size at refwave
    bet = p[3]              # Thin disk wavelength power-law scaling
    if hascontlag:
        lags = np.zeros(nlc)
        wids = np.zeros(nlc)
        scales = np.ones(nlc)
        for i in xrange(1, nlc): # Get values needed for lnlikefn_spear
            lags[i] = thin_disk_func(alph, bet, lwaves[i], refwave)
            wids[i] = p[4+(i-1)*2]   
            scales[i] = p[5+(i-1)*2]
        return(sigma, tau, lags, wids, scales, alph, bet)
    else:
        llags = np.zeros(nlc-1)
        lwids = np.zeros(nlc-1)
        lscales = np.ones(nlc-1)
        for i in xrange(nlc-1):
            llags[i] = thin_disk_func(alph, bet, lwaves[i + 1], refwave)
            lwids[i] = p[4+i*2]
            lscales[i] = p[5+i*2]
        return(sigma, tau, llags, lwids, lscales, alph, bet)

def lnpostfn_thindisk_p(p, zydata, bandwaves, ref_wave, conthpd=None, 
                     lagtobaseline=0.3, laglimit=None,
                     set_threading=False, blocksize=10000, set_retq=False,
                     set_verbose=False, tophatminwidth=None, 
                     a_lims = [0., np.inf], b_lims = [0., np.inf], 
                     fixed=None, p_fix=None):
    """ log-posterior function of p.

    Parameters
    ----------
    p: array_like
        Rmap_Model parameters, [log_e(sigma), log_e(tau), alpha, beta, width1, 
        scale1, ..., widthn, scalen]
    zydata: LightCurve object
        Input LightCurve data.
    bandwaves: array_like
    	The effective wavelengths for the photometric bands
    conthpd: ndarray, optional
        Priors on sigma and tau as an ndarray with shape (3, 2),
        np.array([[log_e(sigma_low), log_e(tau_low)],
                  [log_e(sigma_med), log_e(tau_med)],
                  [log_e(sigma_hig), log_e(tau_hig)]])
        where 'low', 'med', and 'hig' are defined as the 68% confidence
        limits around the median. conthpd usually comes in as an attribute
        of the `Cont_Model` object `hpd` (default: None).
    lagtobaseline: float, optional
        Prior on lags. When input lag exceeds lagtobaseline*baseline, a
        logarithmic prior will be applied.
    laglimit: str or list of tuples.
        hard boundaries for the lag searching during MCMC sampling.
        'baseline' means the boundaries are naturally determined by the
        duration of the light curves, or you can set them as a list with
        `nline` of tuples, with each tuple containing the (min, max) pair
        for each single line.
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
    tophatminwidth: float, optional
        Used for prior on tophat transfer function width (prior4).     
    a_lims: list of floats, optional
	The allowed limits for the disk size of the driving light curve 
	in units of light-(time unit of light curve).
    b_lims: list of floats, optional
	The allowed limits for the power law index of the disk scaling 
	as a function of wavelength.    
    fixed: list
        Bit list indicating which parameters are to be fixed during
        minimization, `1` means varying, while `0` means fixed,
        so [1, 1, 0] means fixing only the third parameter, and `len(fixed)`
        equals the number of parameters (default: None, i.e., varying all
        the parameters simultaneously).
    p_fix: list
        parameter list, with p_fix[fixed==0] being fixed.

    
    Returns
    -------
    retval: float (set_retq is False) or list (set_retq is True)
        if `retval` returns a list, then it contains the full posterior info
        as a list of [log_posterior, chi2_component, det_component,
        DC_penalty, correction_to_the_mean].

    """
    if fixed is not None:
        # fix parameters during inference.
        fixed = np.asarray(fixed)
        p_fix = np.asarray(p_fix)
        p = np.asarray(p)
        p = p * fixed + p_fix * (1. - fixed)
    # unpack the parameters from p
    sigma, tau, llags, lwids, lscales, alpha, beta = unpackthindiskpar(p, 
                              zydata.nlc, hascontlag=False, lwaves=bandwaves,
                              refwave = ref_wave)
    if set_retq:
        vals = list(lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales,
                              set_retq=True, set_verbose=set_verbose,
                              set_threading=set_threading,
                              blocksize=blocksize))
    else:
        logl = lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales,
                              set_retq=False, set_verbose=set_verbose,
                              set_threading=set_threading, blocksize=blocksize)

    # Deal with errors on logl, give large bad value
    if np.isnan(logl):
    	return 6.*my_neg_inf
    # conthpd is in natural log
    if conthpd is not None:
        # for sigma
        if p[0] < conthpd[1,0]:
            prior0 = (p[0] - conthpd[1,0])/(conthpd[1,0]-conthpd[0,0])
        else:
            prior0 = (p[0] - conthpd[1,0])/(conthpd[2,0]-conthpd[1,0])
        # for tau
        if p[1] < conthpd[1,1]:
            prior1 = (p[1] - conthpd[1,1])/(conthpd[1,1]-conthpd[0,1])
        else:
            prior1 = (p[1] - conthpd[1,1])/(conthpd[2,1]-conthpd[1,1])
    else:
        prior0 = 0.0
        prior1 = 0.0
    # for each lag
    prior2 = 0.0
    prior3 = 0.0    # Don't let alpha or beta go negative or outside range
    prior4 = 0.0    # Don't let width go less than specified time units 
    prior5 = 0.0    # Don't let scales go negative
    prior6 = 0.0    # Don't let logsigma go above ceiling, below floor
    prior7 = 0.0    # Don't let logtau go above ceiling, below floor
    for i in xrange(zydata.nlc-1):
        if lagtobaseline < 1.0:
            if np.abs(llags[i]) > lagtobaseline*zydata.rj:
                # penalize long lags when they are larger than 0.3 times the
                # baseline, as it is too easy to fit the model with
                # non-overlapping signals in the light curves.
                prior2 += np.log(np.abs(llags[i])/(lagtobaseline*zydata.rj))
        # penalize long lags to be impossible
        if laglimit is not None:
            if llags[i] > laglimit[i][1] or llags[i] < laglimit[i][0]:
                # try not stack priors
                prior2 = my_pos_inf
    # Add penalty for alpha, beta
    if (alpha <= a_lims[0]) or (beta <= b_lims[0]) or \
       (alpha >= a_lims[1]) or (beta >= b_lims[1]): 
    	prior3 = my_pos_inf
    # Add penalty for tophad widths if that parameter is provided
    if tophatminwidth is not None:
        if (lwids <= tophatminwidth).any():
    	    prior4 = my_pos_inf
    # Add penalty for zero scale
    if (lscales <= 0.).any():
    	prior5 = my_pos_inf
    if (sigma <= sigma_floor) or (sigma >= sigma_ceiling):
        prior6 = my_pos_inf
    if (tau <= tau_floor) or (tau >= tau_ceiling):
        prior7 = my_pos_inf
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1)**2. - prior2 - prior3 - prior4 \
        - prior5 - prior6 - prior7
    if set_retq:
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else:
        logp = logl + prior
        return(logp)

class Disk_Model(object):
    def __init__(self, zydata = None, effwave = None, tophatminwidth = None, 
                 alpha_lims = [0., np.inf], beta_lims = [0., np.inf]):
        """ Disk Model object.

        Parameters
        ----------
        zydata: LightCurve object, necessary
            Light curve data.

	effwave: list-like, necessary
	    Gives the effective wavelengths that the photometric data is at.
	    Assumes the first item in this list is that of the driving zylc.
	    
	tophatminwidth: float, optional
	    The smallest value to allow for the tophat transfer function in
	    the time units supplied by the zydata.
	
	alpha_lims: list of floats, optional
	    The allowed limits for the disk size of the driving light curve 
	    in units of light-(time unit of light curve).
	
	beta_lims: list of floats, optional
	    The allowed limits for the power law index of the disk scaling 
	    as a function of wavelength.    
	    
        """
        if zydata is None:
            raise UsageError("Disk_Model Object requires Light Curve data.")
        else:
            self.zydata = zydata
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]  
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            
            self.names = zydata.names
            
            self.tophatminwidth = None
            self.alpha_lims = np.asarray(alpha_lims, dtype = float)
            self.beta_lims = np.asarray(beta_lims, dtype = float)
            
            if len(effwave) != self.nlc:
            	raise UsageError("Number of wavelengths does not match " + 
            	    "number of light curves.")
            self.effwaves = np.array(effwave, dtype=float)
            self.refwave = self.effwaves[0] 
            # Note the code always assumes the driving zylc wavelength is the 
            #     first entry in self.effwaves, skipping it when necessary
            # number of parameters = 4 globals, then 2 more for each zylc
            self.ndim = 4 + (self.nlc-1)*2    
            self.vars = ["sigma", "tau", "alpha", "beta"]  # always used params
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$", r"$\alpha$", 
                            r"$\beta$"]
            for i in xrange(1, self.nlc):
                self.vars.append("_".join(["wid",   self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append("".join(
                    [r"$w_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))
                self.texs.append("".join(
                    [r"$s_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))

    def __call__(self, p, **lnpostparams):
        """ Calculate the posterior value given one parameter set `p`.
        See `lnpostfn_thindisk_p` for doc.

        Parameters
        ----------
        p: array_like
        Rmap_Model parameters, [log_e(sigma), log_e(tau), alpha, beta, width1, 
            scale1, ..., widthn, scalen]

        lnpostparams: kwargs
            Keyword arguments for `lnpostfn_thindisk_p`.

        Returns
        -------
        retval: float (set_retq is False) or list (set_retq is True)
            if `retval` returns a list, then it contains the full posterior info
            as a list of [log_posterior, chi2_component, det_component,
            DC_penalty, correction_to_the_mean].

        """
        return(lnpostfn_thindisk_p(p, self.zydata, self.effwaves, self.refwave,
               **lnpostparams))



    def do_mcmc(self, conthpd=None, lagtobaseline=0.3, laglimit="baseline",
                nwalkers=100, nburn=100, nchain=100, threads=1, fburn=None,
                fchain=None, flogp=None, set_threading=False, blocksize=10000,
                set_verbose=True, fixed=None, p_fix=None):
        """ Run MCMC sampling over the parameter space.

        Parameters
        ----------
        conthpd: ndarray, optional
            Priors on sigma and tau as an ndarray with shape (3, 2),
            np.array([[log_e(sigma_low), log_e(tau_low)],
                      [log_e(sigma_med), log_e(tau_med)],
                      [log_e(sigma_hig), log_e(tau_hig)]])
            where 'low', 'med', and 'hig' are defined as the 68% confidence
            limits around the median. conthpd usually comes in as an attribute
            of the `Cont_Model` object `hpd` (default: None).
        lagtobaseline: float, optional
            Prior on lags. When input lag exceeds lagtobaseline*baseline, a
            logarithmic prior will be applied.
        laglimit: str or list of tuples.
            Hard boundaries for the lag searching during MCMC sampling.
            'baseline' means the boundaries are naturally determined by the
            duration of the light curves, or you can set them as a list
            with `nline` of tuples, with each tuple containing the (min, max)
            pair for each single line.
        nwalker: integer, optional
            Number of walkers for `emcee` (default: 100).
        nburn: integer, optional
            Number of burn-in steps for `emcee` (default: 50).
        nchain: integer, optional
            Number of chains for `emcee` (default: 50).
        thread: integer
            Number of threads (default: 1).
        fburn: str, optional
            filename for burn-in output (default: None).
        fchain: str, optional
            filename for MCMC chain output (default: None).
        flogp: str, optional
            filename for logp output (default: None).
        set_threading: bool, optional
            True if you want threading in filling matrix. It conflicts with the
            'threads' option in Rmap_Model.run_mcmc (default: False).
        blocksize: int, optional
            Maximum matrix block size in threading (default: 10000).
        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).   
        fixed: list
            Bit list indicating which parameters are to be fixed during
            minimization, `1` means varying, while `0` means fixed,
            so [1, 1, 0] means fixing only the third parameter, and `len(fixed)`
            equals the number of parameters (default: None, i.e., varying all
            the parameters simultaneously).
        p_fix: list
            parameter list, with p_fix[fixed==0] being fixed.
    
        """
        
        # Print statements for knowledge of user.  These values can be changed in this script.
        if (threads > 1 and (not set_threading)):
            if set_verbose:
                print("run parallel chains of number %2d " % threads)
        elif (threads == 1):
            if set_verbose:
                if set_threading:
                    print("run single chain in submatrix blocksize %10d " %
                          blocksize)
                else:
                    print("run single chain without subdividing matrix ")
        else:
            raise InputError("conflicting set_threading and threads setup")
        if laglimit == "baseline":
            laglimit = [[-self.rj, self.rj],]*(self.nlc-1)
        elif len(laglimit) != (self.nlc - 1):
            raise InputError(
                "laglimit should be a list of lists matching number of lines")
        # generate array of random numbers
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
 
        # initialize arrays
        
        
        if conthpd is None:
            #p0[:, 0] += np.log(self.cont_std)-0.5*np.random.rand(nwalkers)
            #p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))- \
            #            0.5*np.random.rand(nwalkers)
            p0[:, 0] = np.random.uniform(low = logsigma_floor, 
                       high = logsigma_ceiling, size = nwalkers)
            p0[:, 1] = np.random.uniform(low = logtau_floor, 
                       high = logtau_ceiling, size = nwalkers)           
        else:
            p0[:, 0] += conthpd[1,0]-0.5*np.random.rand(nwalkers)
            p0[:, 1] += conthpd[1,1]-0.5*np.random.rand(nwalkers)
        # Begin lwids at larger than the cadence, and apply min if supplied
        for i in range(0, self.nlc-1):
            p0[:, 4 + 2*i] *= 2.*self.cont_cad # Scatter widths around cadence
            if self.tophatminwidth is not None:
                p0[:, 4 + 2*i] += self.tophatminwidth
        # Reset alpha and beta initialization if limits were supplied
        if ~(np.isinf(self.alpha_lims).any() or 
           np.isinf(self.beta_lims).any()):
            p0[:, 2] = np.random.uniform(low = self.alpha_lims[0], 
                          high = self.alpha_lims[1], size = nwalkers)
            p0[:, 3] = np.random.uniform(low = self.beta_lims[0], 
                          high = self.beta_lims[1], size = nwalkers)
        # Go to town!
        if set_verbose:
            print("start burn-in")
            if conthpd is None:
                print("no priors on sigma and tau")
            else:
                print("using priors on sigma and tau from continuum fitting")
                print(np.exp(conthpd))
            if lagtobaseline < 1.0:
                print("penalize lags longer than %3.2f of the baseline" %
                      lagtobaseline)
            else:
                print("no penalizing long lags, but within the baseline")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"
                  % (nburn, nwalkers, nburn*nwalkers))
        # initialize the ensemble sampler, proceed to burn in
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_thindisk_p,
                                  args=(self.zydata, self.effwaves, 
                                  self.refwave, conthpd, lagtobaseline,
                                  laglimit, set_threading, blocksize, False,
                                  False, self.tophatminwidth, self.alpha_lims, 
                                  self.beta_lims, fixed, p_fix), 
                                  threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)

        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            if fixed is not None:
                # modify flatchain
                for i in range(self.ndim):
                    if fixed[i] == 0:
                        sampler.flatchain[:, i] = p_fix[i]
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler, lags after burn-in
        sampler.reset()
        self.lags = list()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        if fixed is not None:
                # modify flatchain
                for i in range(self.ndim):
                    if fixed[i] == 0:
                        sampler.flatchain[:, i] = p_fix[i]
        af = sampler.acceptance_fraction
        # Turn lags into array from list for printing
        self.lags = np.asarray(self.lags, dtype = float)
        if set_verbose:
            print("acceptance fractions are")
            print(" ".join([format(r, "3.2f") for r in af]))
        if fchain is not None:
            if set_verbose:
                print("save MCMC chains to %s" % fchain)
            np.savetxt(fchain, sampler.flatchain)
        if flogp is not None:
            if set_verbose:
                print("save logp of MCMC chains to %s" % flogp)
            np.savetxt(flogp, np.ravel(sampler.lnprobability), fmt='%16.8f')
        # make chain an attritue
        self.flatchain = sampler.flatchain
        self.flatchain_whole = np.copy(self.flatchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        self.logp = np.ravel(sampler.lnprobability)
        self.logp_whole = np.copy(self.logp)
        self.get_bfp()

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """ Do an optimization to find the Maximum a Posterior estimates.
        See `lnpostfn_thindisk_p` for doc.

        Parameters
        ----------
        p_ini: array_like
            Rmap_Model parameters, [log_e(sigma), log_e(tau), alpha, beta, 
            wid1, scale1,...,widn, scalen]

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1, 1, 1, 1, ...] means sigma is fixed while others
            are varying. fixed=[1, 1, 1, 1, 1, ...] is equivalent to
            fixed=None (default: None).

        lnpostparams: kwargs
            Kewword arguments for `lnpostfn_thindisk_p`.

        Returns
        -------
        p_bst: array_like
            Best-fit parameters.

        l: float
            The maximum log-posterior.

        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        set_retq = lnpostparams.pop("set_retq", False)
        if set_retq is True:
            raise InputError("set_retq has to be False")
        p_ini = np.asarray(p_ini)
        
        if fixed is not None:
            fixed = np.asarray(fixed)
            func = lambda _p: -lnpostfn_thindisk_p(_p*fixed+p_ini*(1.-fixed),
                                 self.zydata, self.effwaves, self.refwave, 
                                 **lnpostparams)
        else:
            func = lambda _p: -lnpostfn_thindisk_p(_p,
                                                self.zydata, self.effwaves, self.refwave, **lnpostparams)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
	
        sigma, tau, llags, lwids, lscales, alpha, beta = unpackthindiskpar(p_bst, self.nlc,
            hascontlag=False, lwaves=self.effwaves, refwave = self.refwave)
        if len(llags) == self.nlc:
            j = 1
        else:
            j = 0
        
        if set_verbose:
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f alpha %8.3f beta %8.3f" % (sigma, tau, alpha, beta))
            for i in xrange(self.nlc-1):
                ip = 4+i*2
                print("%s %8.3f %s %8.3f" % (
                    self.vars[ip+0], lwids[j],
                    self.vars[ip+1], lscales[j],
                    ))
                j = j + 1
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def get_hpd(self, set_verbose=True):
        """ Get the 68% percentile range of each parameter to self.hpd.

        Parameters
        ----------
        set_verbose: bool, optional
            True if you want verbosity (default: True).

        """
        hpd = _get_hpd(self.ndim, self.flatchain)
        for i in xrange(self.ndim):
            if set_verbose:
                print("HPD of %s" % self.vars[i])
                if i < 2:
                    print("low: %8.3f med %8.3f hig %8.3f" %
                          tuple(np.exp(hpd[:,i])))
                else:
                    print("low: %8.3f med %8.3f hig %8.3f" % tuple(hpd[:,i]))
        # register hpd to attr
        self.hpd = hpd

    def get_bfp(self):
        self.bfp = _get_bfp(self.flatchain, self.logp)

    def show_hist(self, bins=100, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins for parameters except for 'lag' (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).

        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(14, 2.8*(self.nlc+1)))
        for i in xrange(2):
            ax = fig.add_subplot(self.nlc + 1,2,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        for j in xrange(2,4):
            ax = fig.add_subplot(self.nlc + 1,2,j+1)
            ax.hist(self.flatchain[:,j], bins)
            ax.set_xlabel(self.texs[j])
            ax.set_ylabel("N")
        for k in xrange(self.nlc-1):
            for m in xrange(4+k*2, 6+k*2):
                ax = fig.add_subplot(self.nlc + 1, 2, m+1)
                #ax = fig.add_subplot(self.nlc,3,i+1+1)
                ax.hist(self.flatchain[:,m], bins)
                ax.set_xlabel(self.texs[m])
                ax.set_ylabel("N")
        plt.tight_layout()
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def restore_chain(self):
        """ Restore chain after `break_chain`.
        """
        self.flatchain = np.copy(self.flatchain_whole)
        if hasattr(self, "logp"):
            self.logp = np.copy(self.logp_whole)

    def load_chain(self, fchain, flogp=None, set_verbose=True):
        """ Load stored MCMC chain.

        Parameters
        ----------
        fchain: string
            Name for the chain file.

        set_verbose: bool, optional
            True if you want verbosity (default: True).
        """
        if set_verbose:
            print("load MCMC chain from %s" % fchain)
        self.flatchain = np.loadtxt(fchain)
        print "Gen from text complete"
        self.flatchain_whole = np.copy(self.flatchain)
        print "flatchain complete"
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.loadtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

    def get_qlist(self, p_bst):
        """ get the best-fit linear responses.

        Parameters
        ----------
        p_bst: list
            best-fit parameters.
        """
        self.qlist = lnpostfn_thindisk_p(p_bst, self.zydata, self.effwaves, self.refwave, set_retq=True,
                                      set_verbose=False)[4]
    
    
    def do_pred(self, p_bst=None, fpred=None, dense=10, set_overwrite=True):
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
        if p_bst is None and hasattr(self, "bfp"):
            p_bst = self.bfp
        self.get_qlist(p_bst)
        sigma, tau, llags, lwids, lscales = unpackthindiskpar(
            p_bst, self.zydata.nlc, lwaves = self.effwaves, 
            refwave = self.refwave, hascontlag=True)
        # update qlist
        self.zydata.update_qlist(self.qlist)
        # initialize PredictRmap object
        P = PredictRmap(zydata=self.zydata, sigma=sigma, tau=tau,
                        lags=llags, wids=lwids, scales=lscales)
        nwant = dense*self.cont_npt
        jwant0 = self.jstart - 0.1*self.rj
        jwant1 = self.jend + 0.1*self.rj
        jwant = np.linspace(jwant0, jwant1, nwant)
        zylclist_pred = []
        for i in xrange(self.nlc):
            iwant = np.ones(nwant)*(i+1)
            mve, var = P.mve_var(jwant, iwant)
            sig = np.sqrt(var)
            zylclist_pred.append([jwant, mve, sig])
        zydata_pred = LightCurve(zylclist_pred)
        if fpred is not None:
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        return(zydata_pred)
