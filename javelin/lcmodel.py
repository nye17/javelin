# Last-modified: 28 Apr 2014 05:06:35

# generic packages
import numpy as np
# np.seterr(all='raise')
from scipy.optimize import fmin
import matplotlib.pyplot as plt
# internal packages
from cholesky_utils import cholesky, chosolve_from_tri, chodet_from_tri
from zylc import LightCurve
from cov import get_covfunc_dict
from spear import spear, spear_threading
from predict import (PredictSignal, PredictRmap, PredictPmap, PredictSPmap,
                     PredictSCmap)
from gp import FullRankCovariance, NearlyFullRankCovariance
from err import InputError, UsageError
from emcee import EnsembleSampler
from graphic import figure_handler
from copy import copy

my_neg_inf = float(-1.0e+300)
my_pos_inf = float(+1.0e+300)

tau_floor = 1.e-6
tau_ceiling = 1.e+5
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


__all__ = ['Cont_Model', 'Rmap_Model', 'Pmap_Model', 'SPmap_Model',
           'SCmap_Model']


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
    # C_p is a nested list for some reason [[C_p]], so isscalar is bad
    # if np.isscalar(C_p):
    if zydata.issingle:
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
    # print d[0][0]
    # print d[4]
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


# ---------------------------------
# Cont_Model: Continuum Variability

def unpacksinglepar(p, covfunc="drw", uselognu=False):
    """ Internal Function: Unpack the physical parameters from input 1-d
    array for single mode.
    """
    if p[0] > logsigma_ceiling:
        sigma = sigma_ceiling
    elif p[0] < logsigma_floor:
        sigma = sigma_floor
    else:
        sigma = np.exp(p[0])
    if p[1] > logtau_ceiling:
        tau = tau_ceiling
    elif p[1] < logtau_floor:
        tau = tau_floor
    else:
        tau = np.exp(p[1])
    if covfunc == "drw":
        nu = None
    elif uselognu:
        if p[2] < lognu_floor:
            nu = nu_floor
        elif p[2] > lognu_ceiling:
            nu = nu_ceiling
        else:
            nu = np.exp(p[2])
    else:
        nu = p[2]
    return(sigma, tau, nu)


def lnpostfn_single_p(p, zydata, covfunc, taulimit=None, set_prior=True,
                      conthpd=None, uselognu=False, rank="Full",
                      set_retq=False, set_verbose=False):
    """ Calculate the log posterior for parameter set `p`.

    Parameters
    ----------
    p: list
        Parameter list.
    zydata: LightCurve
        Input LightCurve data.
    covfunc: str
        name of the covariance function.
    taulimit: tuple
        lower and upper boundary for tau.
    set_prior: bool, optional
        Turn on/off priors that are pre-defined in `lnpostfn_single_p`
        (default: True).
    conthpd: ndarray, optional
        Priors on sigma and tau as an ndarray with shape (3, 2),
        np.array([[log(sigma_low), log(tau_low)],
                  [log(sigma_med), log(tau_med)],
                  [log(sigma_hig), log(tau_hig)]])
        where 'low', 'med', and 'hig' are defined as the 68% confidence
        limits around the median. Here it is only used if the `covfunc` is
        '(w)kepler2_exp'.
    uselognu: bool, optional
        Whether to use lognu instead of nu (default: False).
    rank: str, optional
        Type of covariance matrix rank, "Full" or "NearlyFull" (
        default: "Full").
    set_retq: bool, optional
        Whether to return all the components of the posterior (default: False).
    set_verbose: bool, optional
        Turn on/off verbose mode (default: True).

    Returns
    -------
    retval: float (set_retq is False) or list (set_retq is True)
        if `retval` returns a list, then it contains the full posterior info
        as a list of [log_posterior, chi2_component, det_component, DC_penalty,
        correction_to_the_mean].

    """
    sigma, tau, nu = unpacksinglepar(p, covfunc, uselognu=uselognu)
    # log-likelihood
    if set_retq:
        vals = list(lnlikefn_single(zydata, covfunc=covfunc, rank=rank,
                                    sigma=sigma, tau=tau, nu=nu, set_retq=True,
                                    set_verbose=set_verbose))
    else:
        logl = lnlikefn_single(zydata, covfunc=covfunc, rank=rank, sigma=sigma,
                               tau=tau, nu=nu, set_retq=False,
                               set_verbose=set_verbose)
    # prior
    prior = 0.0
    if set_prior:
        if covfunc == "kepler2_exp" or covfunc == "wkepler2_exp":
            if conthpd is None:
                # raise RuntimeError("kepler2_exp prior requires conthpd")
                print("Warning: (w)kepler2_exp prior requires conthpd")
            else:
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
                # final
                prior += -0.5*(prior0*prior0+prior1*prior1)
        else:
            prior += - np.log(sigma)
            if tau > zydata.cont_cad:
                prior += - np.log(tau/zydata.cont_cad)
            elif tau < 0.001:
                # 86.4 seconds if input is in days
                prior += my_neg_inf
            else:
                prior += - np.log(zydata.cont_cad/tau)
    if taulimit is not None:
        if tau < taulimit[0] or tau > taulimit[1]:
            prior += my_neg_inf
    # combine prior and log-likelihood
    if set_retq:
        vals[0] = vals[0] + prior
        vals.append(prior)
        return(vals)
    else:
        logp = logl + prior
        return(logp)


def lnlikefn_single(zydata, covfunc="drw", rank="Full", set_retq=False,
                    set_verbose=False, **covparams):
    """ internal function to calculate the log likelihood,
    see `lnpostfn_single_p` for doc.  """
    covfunc_dict = get_covfunc_dict(covfunc, **covparams)
    sigma = covparams.pop("sigma")
    tau = covparams.pop("tau")
    nu = covparams.pop("nu", None)
    # set up covariance function
    if (sigma <= 0.0 or tau <= 0.0):
        return(_exit_with_retval(
            zydata.nlc, set_retq,
            errmsg="Warning: illegal input of parameters",
            set_verbose=set_verbose))
    if covfunc == "pow_exp":
        if nu <= 0.0 or nu >= 2.0:
            return(_exit_with_retval(zydata.nlc, set_retq,
                   errmsg="Warning: illegal input of parameters in nu",
                   set_verbose=set_verbose))
    elif covfunc == "matern":
        if nu <= 0.0:
            return(_exit_with_retval(zydata.nlc, set_retq,
                   errmsg="Warning: illegal input of parameters in nu",
                   set_verbose=set_verbose))
        if nu < 0.0 or nu >= 1.0:
            return(_exit_with_retval(zydata.nlc, set_retq,
                   errmsg="Warning: illegal input of parameters in nu",
                   set_verbose=set_verbose))
    elif covfunc == "kepler2_exp" or covfunc == "wkepler2_exp":
        # here nu is the cutoff time scale
        if nu < 0.0 or nu >= tau:
            return(_exit_with_retval(zydata.nlc, set_retq,
                   errmsg="Warning: illegal input of parameters in nu",
                   set_verbose=set_verbose))
    # test sigma
    # choice of ranks
    if rank == "Full":
        # using full-rank
        C = FullRankCovariance(**covfunc_dict)
    elif rank == "NearlyFull":
        # using nearly full-rank
        C = NearlyFullRankCovariance(**covfunc_dict)
    else:
        raise InputError("No such option for rank "+rank)
    # cholesky decompose S+N so that U^T U = S+N = C
    # using intrinsic method of C without explicitly writing out cmatrix
    try:
        U = C.cholesky(zydata.jarr, observed=False, nugget=zydata.varr)
    except:
        return(_exit_with_retval(zydata.nlc, set_retq,
               errmsg="Warning: non positive-definite covariance C #5",
               set_verbose=set_verbose))
    # print "test U"
    # print U[:6, :6]
    # print U[5, 5]
    # calculate RPH likelihood
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)


class Cont_Model(object):
    def __init__(self, zydata=None, covfunc="drw"):
        """ Cont Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Input LightCurve data, a null input means that `Cont_Model` will
            be loading existing chains (default: None).

        covfunc: str, optional
            Name of the covariance function for the continuum (default: drw)

        """
        self.zydata = zydata
        self.covfunc = covfunc
        if zydata is None:
            pass
        else:
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_cad_min = zydata.cont_cad_min
            self.cont_cad_max = zydata.cont_cad_max
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            self.names = zydata.names
        self.vars = ["sigma", "tau"]
        self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
        if covfunc == "drw":
            self.uselognu = False
            self.ndim = 2
        elif covfunc == "matern" or covfunc == "kepler2_exp" or covfunc == "wkepler2_exp":
            self.uselognu = True
            self.ndim = 3
            self.vars.append("nu")
            self.texs.append(r"$\log\,\nu$")
        else:
            self.uselognu = False
            self.ndim = 3
            self.vars.append("nu")
            self.texs.append(r"$\nu$")

    def __call__(self, p, **lnpostparams):
        """ Calculate the posterior value given one parameter set `p`.
        See `lnpostfn_single_p` for doc.
        """
        return(lnpostfn_single_p(p, self.zydata, covfunc=self.covfunc,
                                 uselognu=self.uselognu, **lnpostparams))

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """
        Maximum A Posterior minimization. See `lnpostfn_single_p` for doc.

        Parameters
        ----------
        p_ini: list
            Initial guess for the parameters.
        fixed: list
            Bit list indicating which parameters are to be fixed during
            minimization, `1` means varying, while `0` means fixed,
            so [1, 1, 0] means fixing only the third parameter, and `len(fixed)`
            equals the number of parameters (default: None, i.e., varying all
            the parameters simultaneously).
        lnpostparams: kwargs
            kwargs for `lnpostfn_single_p`.
        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        set_retq = lnpostparams.pop("set_retq",    False)
        taulimit = lnpostparams.pop("taulimit",  None)
        set_prior = lnpostparams.pop("set_prior",   True)
        rank = lnpostparams.pop("rank",       "Full")
        conthpd = lnpostparams.pop("conthpd",     None)
        if set_retq is True:
            raise InputError("set_retq has to be False")
        p_ini = np.asarray(p_ini)
        if fixed is not None:
            fixed = np.asarray(fixed)
            func = lambda _p: -lnpostfn_single_p(
                _p*fixed+p_ini*(1.-fixed), self.zydata, self.covfunc,
                taulimit=taulimit,
                set_prior=set_prior,
                conthpd=conthpd,
                uselognu=self.uselognu,
                rank=rank,
                set_retq=False,
                set_verbose=set_verbose
                )
        else:
            func = lambda _p: -lnpostfn_single_p(
                _p, self.zydata, self.covfunc,
                taulimit=taulimit,
                set_prior=set_prior,
                conthpd=conthpd,
                uselognu=self.uselognu,
                rank=rank,
                set_retq=False,
                set_verbose=set_verbose
                )
        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, nu = unpacksinglepar(p_bst, covfunc=self.covfunc,
                                         uselognu=self.uselognu)
        if set_verbose:
            print("Best-fit parameters are:")
            print("sigma %8.3f tau %8.3f" % (sigma, tau))
            if nu is not None:
                print("nu %8.3f" % nu)
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def do_grid1d(self, p_ini, fixed, rangex, dx, fgrid1d, **lnpostparams):
        """ Minimization over a 1D grid. See `lnpostfn_single_p` for doc.

        Parameters
        ----------
        p_ini: list
            Initial guess for the parameters.
        fixed: list
            Bit list indicating which parameters are to be fixed during
            minimization, `1` means varying, while `0` means fixed, so [1, 1, 0]
            means fixing only the third parameter, and `len(fixed)` equals the
            number of parameters (default: None, i.e., varying all the
            parameters simultaneously).
        rangex: tuple
            range of `x`, i.e., (xmin, xmax)
        dx: float
            bin size in `x`.
        fgrid1d: str
            filename for the output.
        lnpostparams: kwargs
            kwargs for `lnpostfn_single_p`.

        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        xs = np.arange(rangex[0], rangex[-1]+dx, dx)
        fixed = np.asarray(fixed)
        nfixed = np.sum(fixed == 0)
        if nfixed != 1:
            raise InputError("wrong number of fixed pars ")
        f = open(fgrid1d, "w")
        for x in xs:
            _p_ini = p_ini*fixed + x*(1.-fixed)
            _p, _l = self.do_map(_p_ini, fixed=fixed, **lnpostparams)
            _line = "".join([format(_l, "20.10g"),
                             " ".join([format(r, "10.5f") for r in _p]), "\n"])
            f.write(_line)
            f.flush()
        f.close()
        if set_verbose:
            print("saved grid1d result to %s" % fgrid1d)

    def do_grid2d(self, p_ini, fixed, rangex, dx, rangey, dy, fgrid2d,
                  **lnpostparams):
        """ Minimization over a 2D grid. See `lnpostfn_single_p` for doc.

        Parameters
        ----------
        p_ini: list
            Initial guess for the parameters.
        fixed: list
            Bit list indicating which parameters are to be fixed during
            minimization, `1` means varying, while `0` means fixed,
            so [1, 1, 0] means fixing only the third parameter,
            and `len(fixed)` equals the number of parameters (default:
                None, i.e., varying all the parameters simultaneously).
        rangex: tuple
            range of `x`, i.e., (xmin, xmax)
        dx: float
            bin size in `x`.
        rangey: tuple
            range of `y`, i.e., (ymin, ymax)
        dy: float
            bin size in `y`.
        fgrid2d: str
            filename for the output.
        lnpostparams: kwargs
            kwargs for `lnpostfn_single_p`.

        """
        fixed = np.asarray(fixed)
        set_verbose = lnpostparams.pop("set_verbose", True)
        xs = np.arange(rangex[0], rangex[-1]+dx, dx)
        ys = np.arange(rangey[0], rangey[-1]+dy, dy)
        nfixed = np.sum(fixed == 0)
        if nfixed != 2:
            raise InputError("wrong number of fixed pars ")
        posx, posy = np.nonzero(1-fixed)[0]
        dimx, dimy = len(xs),len(ys)
        header = " ".join(["#", str(posx), str(posy), str(dimx),
                           str(dimy), "\n"])
        print(header)
        f = open(fgrid2d, "w")
        f.write(header)
        for x in xs:
            for y in ys:
                _p_ini = p_ini*fixed
                _p_ini[posx] = x
                _p_ini[posy] = y
                _p, _l = self.do_map(_p_ini, fixed=fixed, **lnpostparams)
                _line = "".join([format(_l, "20.10g"),
                                 " ".join([format(r, "10.5f") for r in _p]),
                                 "\n"])
                f.write(_line)
                f.flush()
        f.close()
        if set_verbose:
            print("saved grid2d result to %s" % fgrid2d)

    def read_logp_map(self, fgrid2d, set_verbose=True):
        """ Read the output from `do_grid2d`.

        Parameters
        ----------
        fgrid2d: str
            filename.
        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).

        Returns
        -------
        retdict: dict
            Grid returned as a dict.

        """
        f = open(fgrid2d, "r")
        posx, posy, dimx, dimy = [
            int(r) for r in f.readline().lstrip("#").split()]
        if set_verbose:
            print("grid file %s is registered for" % fgrid2d)
            print("var_x = %10s var_y = %10s" % (self.vars[posx],
                                                 self.vars[posy]))
            print("dim_x = %10d dim_y = %10d" % (dimx, dimy))
        if self.covfunc != "drw":
            logp, sigma, tau, nu = np.genfromtxt(
                f, unpack=True, usecols=(0,1,2,3))
        else:
            logp, sigma, tau = np.genfromtxt(f, unpack=True, usecols=(0,1,2))
        f.close()
        retdict = {
            'logp': logp.reshape(dimx, dimy).T,
            'sigma': sigma.reshape(dimx, dimy).T,
            'tau': tau.reshape(dimx, dimy).T,
            'nu': None,
            'posx': posx,
            'posy': posy,
            'dimx': dimx,
            'dimy': dimy,
        }
        if self.covfunc != "drw":
            retdict['nu'] = nu.reshape(dimx, dimy).T
        return(retdict)

    def show_logp_map(self, fgrid2d, set_normalize=True, vmin=None, vmax=None,
                      set_contour=True, clevels=None, set_verbose=True,
                      figout=None, figext=None):
        """ Display the grid output from `do_grid2d`.

        Parameters
        ----------
        fgrid2d: str
            filename.
        set_normalize: bool, optional
            Whether to normalize the histogram.
        vmin: float, optional
            Minimum value of the histogram.
        set_contour: bool, optional
            Whether to overplot contours (default: True).
        clevels: list, optional
            Contour levels. `clevels` = None will set the levels as if the
            likelihood is for a Gaussian model with two parameters.
        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).
        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).
        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        retdict = self.read_logp_map(fgrid2d, set_verbose=set_verbose)
        x = retdict[self.vars[retdict['posx']]]/ln10
        y = retdict[self.vars[retdict['posy']]]/ln10
        z = retdict['logp']
        if x is None or y is None:
            raise InputError("incompatible fgrid2d file"+fgrid2d)
        xmin,xmax,ymin,ymax = np.min(x),np.max(x),np.min(y),np.max(y)
        extent = (xmin,xmax,ymin,ymax)
        if set_normalize:
            zmax = np.max(z)
            z = z - zmax
        if vmin is None:
            vmin = z.min()
        if vmax is None:
            vmax = z.max()
        ax.imshow(z, origin='lower', vmin=vmin, vmax=vmax,
                  cmap='jet', interpolation="nearest", aspect="auto",
                  extent=extent)
        if set_contour:
            if clevels is None:
                sigma3,sigma2,sigma1 = 11.8/2.0,6.17/2.0,2.30/2.0
                levels = (vmax-sigma1, vmax-sigma2, vmax-sigma3)
            else:
                levels = clevels
            ax.set_autoscale_on(False)
            ax.contour(z,levels, hold='on',colors='k',
                       origin='lower',extent=extent)
        ax.set_xlabel(self.texs[retdict['posx']])
        ax.set_ylabel(self.texs[retdict['posy']])
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def do_mcmc(self, conthpd=None, set_prior=True, taulimit="baseline",
                rank="Full", nwalkers=100, nburn=50, nchain=50, fburn=None,
                fchain=None, flogp=None, threads=1, set_verbose=True):
        """ Run MCMC sampling over the parameter space.

        Parameters
        ----------
        conthpd: ndarray, optional
            Usually the `hpd` array generated from the MCMC chain
            using `Cont_Model` (default: None).
        set_prior: bool, optional
            Turn on/off priors that are predefined in `lnpostfn_single_p` (
            default: True).
        taulimit: tuple
            lower and upper boundary for tau.
        rank: str, optional
            Type of covariance matrix rank, "Full" or "NearlyFull" (
            default: "Full").
        nwalker: integer, optional
            Number of walkers for `emcee` (default: 100).
        nburn: integer, optional
            Number of burn-in steps for `emcee` (default: 50).
        nchain: integer, optional
            Number of chains for `emcee` (default: 50).
        fburn: str, optional
            filename for burn-in output (default: None).
        fchain: str, optional
            filename for MCMC chain output (default: None).
        flogp: str, optional
            filename for logp output (default: None).
        thread: integer
            Number of threads (default: 1).
        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).

        """
        # initialize a multi-dim random number array
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initial values of sigma to be scattering around cont_std
        p0[:, 0] = p0[:, 0] - 0.5 + np.log(self.cont_std)
        # initial values of tau   filling cont_cad : cont_cad + 0.5rj
        # p0[:, 1] = np.log(self.cont_cad + self.rj*0.5*p0[:, 1])
        p0[:, 1] = np.log(2.0 * self.cont_cad + self.rj*0.5*p0[:, 1])
        if self.covfunc == "pow_exp":
            p0[:, 2] = p0[:, 2] * 1.99
        elif self.covfunc == "matern":
            p0[:, 2] = np.log(p0[:, 2] * 5)
        elif self.covfunc == "kepler2_exp" or self.covfunc == "wkepler2_exp":
            # p0[:, 2] = np.log(self.cont_cad * p0[:, 2])
            p0[:, 2] = np.log(2.0 * self.cont_cad * p0[:, 2])
            # p0[:, 2] = np.log(self.cont_cad * p0[:, 2])
            # p0[:, 2] = p0[:, 1] + np.log(0.2)
            # make sure the initial values of tau_cut are smaller than tau_d
        if set_verbose:
            print("start burn-in")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"
                  % (nburn, nwalkers, nburn*nwalkers))
        if taulimit == "baseline":
            taulimit = [self.cont_cad, self.rj]
        sampler = EnsembleSampler(
            nwalkers, self.ndim, lnpostfn_single_p,
            args=(self.zydata, self.covfunc, taulimit, set_prior, conthpd,
                  self.uselognu, rank, False, False), threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset sampler
        sampler.reset()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        af = sampler.acceptance_fraction
        if set_verbose:
            print("acceptance fractions for all walkers are")
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

    def get_hpd(self, set_verbose=True):
        """ Get the 68% percentile range of each parameter to self.hpd.

        Parameters
        ----------
        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).

        """
        hpd = _get_hpd(self.ndim, self.flatchain)
        for i in xrange(self.ndim):
            if set_verbose:
                print("HPD of %s" % self.vars[i])
                if (self.vars[i] == "nu" and (not self.uselognu)):
                    print("low: %8.3f med %8.3f hig %8.3f" % tuple(hpd[:,i]))
                else:
                    print("low: %8.3f med %8.3f hig %8.3f" % tuple(
                        np.exp(hpd[:,i])))
        # register hpd to attr
        self.hpd = hpd

    def get_bfp(self):
        self.bfp = _get_bfp(self.flatchain, self.logp)

    def show_hist(self, bins=100, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).
        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(8, 5))
        for i in xrange(self.ndim):
            ax = fig.add_subplot(1,self.ndim,i+1)
            if (self.vars[i] == "nu" and (not self.uselognu)):
                ax.hist(self.flatchain[:,i], bins)
                if self.covfunc == "kepler2_exp" or self.covfunc == "wkepler2_exp":
                    ax.axvspan(self.cont_cad_min,
                               self.cont_cad, color="g", alpha=0.2)
            else:
                ax.hist(self.flatchain[:,i]/ln10, bins)
                if self.vars[i] == "nu" and (self.covfunc == "kepler2_exp" or self.covfunc == "wkepler2_exp"):
                    ax.axvspan(np.log10(self.cont_cad_min),
                               np.log10(self.cont_cad), color="g", alpha=0.2)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        # plt.get_current_fig_manager().toolbar.zoom()
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def load_chain(self, fchain, flogp=None, set_verbose=True):
        """ Load an existing chain file.

        Parameters
        ----------
        fchain: str
            filename for MCMC chain input.

        set_verbose: bool, optional
            Turn on/off verbose mode (default: True).

        """
        if set_verbose:
            print("load MCMC chain from %s" % fchain)
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.genfromtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

    def break_chain(self, covpar_segments):
        """ Break the chain into different segments.

        Parameters
        ----------
        covpar_segments: list of lists.
            list with length that equals the number of dimensions of the
            parameter space.
        """
        if (len(covpar_segments) != self.ndim):
            print("Error: covpar_segments has to be a list of length %d" %
                  (self.ndim))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        for i, covpar_seq in enumerate(covpar_segments):
            if covpar_seq is None:
                continue
            indx = np.argsort(self.flatchain[:, i])
            imin, imax = np.searchsorted(self.flatchain[indx, i], covpar_seq)
            indx_cut = indx[imin: imax]
            if len(indx_cut) < 10:
                print("Warning: cut too aggressive!")
                return(1)
            self.flatchain = self.flatchain[indx_cut,:]
            if hasattr(self, "logp"):
                self.logp = self.logp[indx_cut]

    def restore_chain(self):
        """ Restore chain after `break_chain`.
        """
        self.flatchain = np.copy(self.flatchain_whole)
        if hasattr(self, "logp"):
            self.logp = np.copy(self.logp_whole)

    def get_qlist(self, p_bst):
        """ get the best-fit linear responses.

        Parameters
        ----------
        p_bst: list
            best-fit parameters.
        """
        self.qlist = lnpostfn_single_p(p_bst, self.zydata, self.covfunc,
                                       uselognu=self.uselognu, rank="Full",
                                       set_retq=True)[4]

    def do_pred(self, p_bst=None, fpred=None, dense=10, rank="Full",
                set_overwrite=True):
        """ Predict light curves using the best-fit parameters.

        Parameters
        ----------
        p_bst: list
            best-fit parameters.
        fpred: str, optional
            filename for saving the predicted light curves.
        dense: integer, optional
            factors by which the desired sampling is compared to the original
            data sampling (default: 10).
        rank: str, optional
            Type of covariance matrix rank, "Full" or "NearlyFull" (
            default: "Full").
        set_overwrite: bool, optional
            Whether to overwrite, if a `fpred` file already exists.

        Returns
        -------
        zypred: LightCurve data.
            Predicted LightCurve.

        """
        if p_bst is None and hasattr(self, "bfp"):
            p_bst = self.bfp
        self.get_qlist(p_bst)
        self.zydata.update_qlist(self.qlist)
        sigma, tau, nu = unpacksinglepar(p_bst, self.covfunc,
                                         uselognu=self.uselognu)
        lcmean = self.zydata.blist[0]
        P = PredictSignal(zydata=self.zydata, lcmean=lcmean, rank=rank,
                          covfunc=self.covfunc, sigma=sigma, tau=tau, nu=nu)
        nwant = dense*self.cont_npt
        jwant0 = self.jstart - 0.1*self.rj
        jwant1 = self.jend + 0.1*self.rj
        jwant = np.linspace(jwant0, jwant1, nwant)
        mve, var = P.mve_var(jwant)
        sig = np.sqrt(var)
        zylclist_pred = [[jwant, mve, sig],]
        zydata_pred = LightCurve(zylclist_pred)
        if fpred is not None:
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        return(zydata_pred)


# ---------------------------------
# Rmap_Model: Spectroscopic RM

def unpackspearpar(p, nlc=None, hascontlag=False):
    """ Internal Function: unpack the physical parameters from input 1-d
    array for spear mode.
    """
    if nlc is None:
        # possible to figure out nlc from the size of p
        nlc = (len(p) - 2)//3 + 1
    sigma = np.exp(p[0])
    tau = np.exp(p[1])
    if hascontlag:
        lags = np.zeros(nlc)
        wids = np.zeros(nlc)
        scales = np.ones(nlc)
        for i in xrange(1, nlc):
            lags[i] = p[2+(i-1)*3]
            wids[i] = p[3+(i-1)*3]
            scales[i] = p[4+(i-1)*3]
        return(sigma, tau, lags, wids, scales)
    else:
        llags = np.zeros(nlc-1)
        lwids = np.zeros(nlc-1)
        lscales = np.ones(nlc-1)
        for i in xrange(nlc-1):
            llags[i] = p[2+i*3]
            lwids[i] = p[3+i*3]
            lscales[i] = p[4+i*3]
        return(sigma, tau, llags, lwids, lscales)


def lnpostfn_spear_p(p, zydata, conthpd=None, lagtobaseline=0.3, laglimit=None,
                     set_threading=False, blocksize=10000, set_retq=False,
                     set_verbose=False):
    """ log-posterior function of p.

    Parameters
    ----------
    p: array_like
        Rmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1,
        ...]
    zydata: LightCurve object
        Input LightCurve data.
    conthpd: ndarray, optional
        Priors on sigma and tau as an ndarray with shape (3, 2),
        np.array([[log(sigma_low), log(tau_low)],
                  [log(sigma_med), log(tau_med)],
                  [log(sigma_hig), log(tau_hig)]])
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

    Returns
    -------
    retval: float (set_retq is False) or list (set_retq is True)
        if `retval` returns a list, then it contains the full posterior info
        as a list of [log_posterior, chi2_component, det_component,
        DC_penalty, correction_to_the_mean].

    """
    # unpack the parameters from p
    sigma, tau, llags, lwids, lscales = unpackspearpar(p, zydata.nlc,
                                                       hascontlag=False)
    if set_retq:
        vals = list(lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales,
                                   set_retq=True, set_verbose=set_verbose,
                                   set_threading=set_threading,
                                   blocksize=blocksize))
    else:
        logl = lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales,
                              set_retq=False, set_verbose=set_verbose,
                              set_threading=set_threading, blocksize=blocksize)
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
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1) - prior2
    if set_retq:
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else:
        logp = logl + prior
        return(logp)


def lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, set_retq=False,
                   set_verbose=False, set_threading=False, blocksize=10000):
    """ Internal function to calculate the log likelihood.
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
    lags[1:] = llags
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
                errmsg="Warning: non positive-definite covariance C #4",
                set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)


class Rmap_Model(object):
    def __init__(self, zydata=None):
        """ Rmap Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        """
        self.zydata = zydata
        if zydata is None:
            pass
        else:
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            self.names = zydata.names
            # number of parameters
            self.ndim = 2 + (self.nlc-1)*3
            self.vars = ["sigma", "tau"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            for i in xrange(1, self.nlc):
                self.vars.append("_".join(["lag",   self.names[i]]))
                self.vars.append("_".join(["wid",   self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append("".join(
                    [r"$t_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))
                self.texs.append("".join(
                    [r"$w_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))
                self.texs.append("".join(
                    [r"$s_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))

    def __call__(self, p, **lnpostparams):
        """ Calculate the posterior value given one parameter set `p`.
        See `lnpostfn_spear_p` for doc.

        Parameters
        ----------
        p: array_like
            Rmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1,
            ...]

        lnpostparams: kwargs
            Keyword arguments for `lnpostfn_spear_p`.

        Returns
        -------
        retval: float (set_retq is False) or list (set_retq is True)
            if `retval` returns a list, then it contains the full posterior info
            as a list of [log_posterior, chi2_component, det_component,
            DC_penalty, correction_to_the_mean].

        """
        return(lnpostfn_spear_p(p, self.zydata, **lnpostparams))

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """ Do an optimization to find the Maximum a Posterior estimates.
        See `lnpostfn_spear_p` for doc.

        Parameters
        ----------
        p_ini: array_like
            Rmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1,
            ...]

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1, 1, 1, 1, ...] means sigma is fixed while others
            are varying. fixed=[1, 1, 1, 1, 1, ...] is equivalent to
            fixed=None (default: None).

        lnpostparams: kwargs
            Kewword arguments for `lnpostfn_spear_p`.

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
            func = lambda _p: -lnpostfn_spear_p(_p*fixed+p_ini*(1.-fixed),
                                                self.zydata, **lnpostparams)
        else:
            func = lambda _p: -lnpostfn_spear_p(_p,
                                                self.zydata, **lnpostparams)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, llags, lwids, lscales = unpackspearpar(
            p_bst, self.zydata.nlc, hascontlag=False)
        if set_verbose:
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f" % (sigma, tau))
            for i in xrange(self.nlc-1):
                ip = 2+i*3
                print("%s %8.3f %s %8.3f %s %8.3f" % (
                    self.vars[ip+0], llags[i],
                    self.vars[ip+1], lwids[i],
                    self.vars[ip+2], lscales[i],
                    ))
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def do_mcmc(self, conthpd=None, lagtobaseline=0.3, laglimit="baseline",
                nwalkers=100, nburn=100, nchain=100, threads=1, fburn=None,
                fchain=None, flogp=None, set_threading=False, blocksize=10000,
                set_verbose=True):
        """ Run MCMC sampling over the parameter space.

        Parameters
        ----------
        conthpd: ndarray, optional
            Priors on sigma and tau as an ndarray with shape (3, 2),
            np.array([[log(sigma_low), log(tau_low)],
                      [log(sigma_med), log(tau_med)],
                      [log(sigma_hig), log(tau_hig)]])
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
        """
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
        # initialize array
        if conthpd is None:
            p0[:, 0] += np.log(self.cont_std)-0.5
            p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        else:
            p0[:, 0] += conthpd[1,0]-0.5
            p0[:, 1] += conthpd[1,1]-0.5
        for i in xrange(self.nlc-1):
            p0[:, 2+i*3] = p0[:,2+i*3]*(laglimit[i][1]-laglimit[i][0]) + \
                laglimit[i][0]
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
        # initialize the ensemble sampler
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_spear_p,
                                  args=(self.zydata, conthpd, lagtobaseline,
                                        laglimit, set_threading, blocksize,
                                        False, False), threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler
        sampler.reset()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        af = sampler.acceptance_fraction
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

    def show_hist(self, bins=100, lagbinsize=1.0, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins for parameters except for 'lag' (default:100).

        lagbinsize: integer, optional
            bin width for 'lag' (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).

        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(14, 2.8*self.nlc))
        for i in xrange(2):
            ax = fig.add_subplot(self.nlc,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        for k in xrange(self.nlc-1):
            for i in xrange(2+k*3, 5+k*3):
                ax = fig.add_subplot(self.nlc,3,i+1+1)
                if np.mod(i, 3) == 2:
                    # lag plots
                    lagbins = np.arange(
                        int(np.min(self.flatchain[:,i])),
                        int(np.max(self.flatchain[:,i]))+lagbinsize, lagbinsize)
                    ax.hist(self.flatchain[:,i], bins=lagbins)
                else:
                    ax.hist(self.flatchain[:,i], bins)
                ax.set_xlabel(self.texs[i])
                ax.set_ylabel("N")
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
        if (len(llag_segments) != self.nlc-1):
            print("Error: llag_segments has to be a list of length %d" %
                  (self.nlc-1))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        for i, llag_seq in enumerate(llag_segments):
            if llag_seq is None:
                continue
            indx = np.argsort(self.flatchain[:, 2+i*3])
            imin, imax = np.searchsorted(self.flatchain[indx, 2+i*3], llag_seq)
            indx_cut = indx[imin: imax]
            self.flatchain = self.flatchain[indx_cut,:]
            if hasattr(self, "logp"):
                self.logp = self.logp[indx_cut]

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
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.genfromtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

    def get_qlist(self, p_bst):
        """ get the best-fit linear responses.

        Parameters
        ----------
        p_bst: list
            best-fit parameters.
        """
        self.qlist = lnpostfn_spear_p(p_bst, self.zydata, set_retq=True,
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
        sigma, tau, lags, wids, scales = unpackspearpar(
            p_bst, self.zydata.nlc, hascontlag=True)
        # update qlist
        self.zydata.update_qlist(self.qlist)
        # initialize PredictRmap object
        P = PredictRmap(zydata=self.zydata, sigma=sigma, tau=tau,
                        lags=lags, wids=wids, scales=scales)
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


# ---------------------------------
# Pmap_Model: Two-Band Spectroscopic RM

def unpackphotopar(p, nlc=2, hascontlag=False):
    """ Unpack the physical parameters from input 1-d array for photo mode.

    Currently only two bands, one on and on off the line emission.

    """
    if nlc != 2:
        raise InputError("Pmap_Model cannot cope with more than two bands yet")
    sigma = np.exp(p[0])
    tau = np.exp(p[1])
    if hascontlag:
        lags = np.zeros(3)
        wids = np.zeros(3)
        scales = np.ones(3)
        # line contribution
        lags[1] = p[2]
        wids[1] = p[3]
        scales[1] = p[4]
        # continuum contribution
        scales[2] = p[5]
        return(sigma, tau, lags, wids, scales)
    else:
        llags = np.zeros(2)
        lwids = np.zeros(2)
        lscales = np.ones(2)
        llags[0] = p[2]
        lwids[0] = p[3]
        lscales[0] = p[4]
        # continuum contribution
        lscales[1] = p[5]
        return(sigma, tau, llags, lwids, lscales)


def lnpostfn_photo_p(p, zydata, conthpd=None, set_extraprior=False,
                     lagtobaseline=0.3, laglimit=None, widtobaseline=0.2,
                     widlimit=None, set_threading=False, blocksize=10000,
                     set_retq=False, set_verbose=False):
    """ log-posterior function of p.

    Parameters
    ----------
    p: array_like
        Pmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1, alpha]
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
    set_extraprior: bool, optional
        DEPRECATED, keep it for backward compatibilit and debugging purposes.
    lagtobaseline: float, optional
        Prior on lags. When input lag exceeds lagtobaseline*baseline, a
        logarithmic prior will be applied.
    laglimit: list of tuples.
        hard boundaries for the lag searching.
    widtobaseline: float, optional
        Prior on wids. When input wid exceeds widtobaseline*baseline, a
        logarithmic prior will be applied.
    widlimit: list of tuples, optional
        hard boundaries for the wid searching.
    set_threading: bool, optional
        True if you want threading in filling matrix. It conflicts with the
        'threads' option in Pmap_Model.run_mcmc (default: False).
    blocksize: int, optional
        Maximum matrix block size in threading (default: 10000).
    set_retq: bool, optional
        Return the value(s) of q along with each component of the
        log-likelihood if True (default: False).
    set_verbose: bool, optional
        True if you want verbosity (default: False).

    """
    # unpack the parameters from p
    sigma, tau, llags, lwids, lscales = unpackphotopar(p, zydata.nlc,
                                                       hascontlag=False)
    if set_retq:
        vals = list(lnlikefn_photo(zydata, sigma, tau, llags, lwids, lscales,
                                   set_retq=True, set_verbose=set_verbose,
                                   set_threading=set_threading,
                                   blocksize=blocksize))
    else:
        logl = lnlikefn_photo(zydata, sigma, tau, llags, lwids, lscales,
                              set_retq=False, set_verbose=set_verbose,
                              set_threading=set_threading, blocksize=blocksize)
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
    if lagtobaseline < 1.0:
        if np.abs(llags[0]) > lagtobaseline*zydata.rj:
            # penalize long lags when larger than 0.3 times the baseline,
            # as it is too easy to fit the model with non-overlapping
            # signals in the light curves.
            prior2 += np.log(np.abs(llags[0])/(lagtobaseline*zydata.rj))
    # penalize long lags to be impossible
    if laglimit is not None:
        if llags[0] > laglimit[0][1] or llags[0] < laglimit[0][0]:
            prior2 += my_pos_inf
    # penalize on extremely large transfer function width
    if widtobaseline < 1.0:
        if np.abs(lwids[0]) > widtobaseline*zydata.rj:
            prior2 += np.log(np.abs(lwids[0])/(widtobaseline*zydata.rj))
    if widlimit is not None:
        if lwids[0] > widlimit[0][1] or lwids[0] < widlimit[0][0]:
            prior2 += my_pos_inf
    # if np.abs(lwids[0]) >= zydata.cont_cad:
        # prior2 += np.log(np.abs(lwids[0])/zydata.cont_cad)
    # else:
        # prior2 += np.log(zydata.cont_cad/np.abs(lwids[0]))
    if set_extraprior:
        # XXX {{{Extra penalizations.
        # penalize on extremely short lags (below median cadence).
        if (np.abs(llags[0]) <= zydata.cont_cad or
                np.abs(llags[0]) <= np.abs(lwids[0])):
            prior2 += my_pos_inf
        # penalize on extremely small line responses (below mean error level).
        if sigma * np.abs(lscales[0]) <= np.mean(zydata.elist[1]):
            prior2 += my_pos_inf
        # }}}
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1) - prior2
    # print p
    # print prior
    if set_retq:
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else:
        logp = logl + prior
        return(logp)


def lnlikefn_photo(zydata, sigma, tau, llags, lwids, lscales, set_retq=False,
                   set_verbose=False, set_threading=False, blocksize=10000):
    """ Log-likelihood function.
    """
    if zydata.issingle:
        raise UsageError("lnlikefn_photo does not work for single mode")
    # impossible scenarios
    if (sigma <= 0.0 or tau <= 0.0 or np.min(lwids) < 0.0 or
            np.min(lscales) < 0.0 or np.max(np.abs(llags)) > zydata.rj):
        return(_exit_with_retval(zydata.nlc, set_retq,
                                 errmsg="Warning: illegal input of parameters",
                                 set_verbose=set_verbose))
    # set_pmap = True
    # fill in lags/wids/scales
    lags = np.zeros(3)
    wids = np.zeros(3)
    scales = np.ones(3)
    lags[1:] = llags[:]
    wids[1:] = lwids[:]
    scales[1:] = lscales[:]
    if set_threading:
        C = spear_threading(zydata.jarr, zydata.jarr, zydata.iarr,
                            zydata.iarr, sigma, tau, lags, wids, scales,
                            set_pmap=True, blocksize=blocksize)
    else:
        C = spear(zydata.jarr, zydata.jarr, zydata.iarr, zydata.iarr, sigma,
                  tau, lags, wids, scales, set_pmap=True)
    # decompose C inplace
    U, info = cholesky(C, nugget=zydata.varr, inplace=True, raiseinfo=False)
    # handle exceptions here
    if info > 0:
        return(_exit_with_retval(
            zydata.nlc, set_retq,
            errmsg="Warning: non positive-definite covariance C #3",
            set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)


class Pmap_Model(object):
    def __init__(self, zydata=None, linename="line"):
        """ Pmap Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        linename: str, optional
            Name of the emission line (default: 'line').

        """
        self.zydata = zydata
        if zydata is None:
            pass
        else:
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            self.names = zydata.names
            # number of parameters
            self.ndim = 6
            self.vars = ["sigma", "tau"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            #
            self.vars.append("_".join(["lag", linename]))
            self.vars.append("_".join(["wid", linename]))
            self.vars.append("_".join(["scale", linename]))
            self.texs.append("".join([r"$t_{", linename, r"}$"]))
            self.texs.append("".join([r"$w_{", linename, r"}$"]))
            self.texs.append("".join([r"$s_{", linename, r"}$"]))
            #
            self.vars.append("alpha")
            self.texs.append(r"$\alpha$")

    def __call__(self, p, **lnpostparams):
        """ Calculate the posterior value given one parameter set `p`.

        Parameters
        ----------
        p: array_like
            Pmap_Model parameters, [log(sigma), log(tau), lag, wid, scale,
            alpha].

        lnpostparams: kwargs
            Kewword arguments for `lnpostfn_photo_p`.

        Returns
        -------
        retval: float (set_retq is False) or list (set_retq is True)
            if `retval` returns a list, then it contains the full posterior info
            as a list of [log_posterior, chi2_component, det_component,
            DC_penalty, correction_to_the_mean].

        """
        return(lnpostfn_photo_p(p, self.zydata, **lnpostparams))

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """ Do an optimization to find the Maximum a Posterior estimates.

        Parameters
        ----------
        p_ini: array_like
            Pmap_Model parameters [log(sigma), log(tau), lag, wid, scale,
            alpha].

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1, 1, 1, 1, 1] means sigma is fixed while others are
            varying. fixed=[1, 1, 1, 1, 1,] is equivalent to fixed=None (
            default: None).

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
            func = lambda _p: -lnpostfn_photo_p(_p*fixed+p_ini*(1.-fixed),
                                                self.zydata, **lnpostparams)
        else:
            func = lambda _p: -lnpostfn_photo_p(_p,
                                                self.zydata, **lnpostparams)
        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, llags, lwids, lscales = unpackphotopar(
            p_bst, self.zydata.nlc, hascontlag=False)
        if set_verbose:
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f" % (sigma, tau))
            print("%s %8.3f %s %8.3f %s %8.3f" % (
                self.vars[2], llags[0], self.vars[3], lwids[0],
                self.vars[4], lscales[0]))
            print("alpha %8.3f" % (lscales[1]))
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def do_mcmc(self, conthpd=None, set_extraprior=False, lagtobaseline=0.3,
                laglimit="baseline", widtobaseline=0.2, widlimit="nyquist",
                nwalkers=100, nburn=100, nchain=100, threads=1, fburn=None,
                fchain=None, flogp=None, set_threading=False, blocksize=10000,
                set_verbose=True):
        """ See `lnpostfn_photo_p` for doc, except for `laglimit` and `widlimit`,
        both of which have different default values ('baseline' / 'nyquist').
        'baseline' means the boundaries are naturally determined by the
        duration of the light curves, and 'nyquist' means the transfer function
        width has to be within two times the typical cadence of light curves.
        """
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
            laglimit = [[-self.rj, self.rj],]
        elif len(laglimit) != 1:
            raise InputError("laglimit should be a list of a single list")
        if widlimit == "nyquist":
            # two times the cadence, resembling Nyquist sampling.
            widlimit = [[0.0, 2.0*self.cont_cad],]
        elif len(widlimit) != 1:
            raise InputError("widlimit should be a list of a single list")
        # generate array of random numbers
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initialize array
        if conthpd is None:
            p0[:, 0] += np.log(self.cont_std)-0.5
            p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        else:
            # XXX stretch the range from (0,1) to ( conthpd[0,0], conthpd[2,0] )
            p0[:, 0] = p0[:, 0] * (conthpd[2,0] - conthpd[0,0]) + conthpd[0,0]
            p0[:, 1] = p0[:, 1] * (conthpd[2,1] - conthpd[0,1]) + conthpd[0,1]
            # old way, just use 0.5 as the 1\sigma width.
            # p0[:, 0] += conthpd[1,0]-0.5
            # p0[:, 1] += conthpd[1,1]-0.5
        p0[:, 2] = p0[:,2]*(laglimit[0][1]-laglimit[0][0]) + laglimit[0][0]
        p0[:, 3] = p0[:,3]*(widlimit[0][1]-widlimit[0][0]) + widlimit[0][0]
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
                print("no penalizing long lags, restrict to < baseline")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"
                  % (nburn, nwalkers, nburn*nwalkers))
        # initialize the ensemble sampler
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_photo_p,
                                  args=(self.zydata, conthpd, set_extraprior,
                                        lagtobaseline, laglimit, widtobaseline,
                                        widlimit, set_threading, blocksize,
                                        False, False), threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler
        sampler.reset()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        af = sampler.acceptance_fraction
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
                    print("low: %8.3f med %8.3f hig %8.3f" %
                          tuple(hpd[:,i]))
        # register hpd to attr
        self.hpd = hpd

    def get_bfp(self):
        self.bfp = _get_bfp(self.flatchain, self.logp)

    def show_hist(self, bins=100, lagbinsize=1.0, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins for parameters except for 'lag' (default:100).

        lagbinsize: integer, optional
            bin width for 'lag' (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).

        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(14, 2.8*self.nlc))
        for i in xrange(2):
            ax = fig.add_subplot(self.nlc,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        # alpha
        ax = fig.add_subplot(self.nlc,3,3)
        ax.hist(self.flatchain[:,5], bins)
        ax.set_xlabel(self.texs[5])
        ax.set_ylabel("N")
        # line
        for i in xrange(2, 5):
            ax = fig.add_subplot(self.nlc,3,i+1+1)
            if np.mod(i, 3) == 2:
                # lag plots
                lagbins = np.arange(int(np.min(self.flatchain[:,i])),
                                    int(np.max(self.flatchain[:,i]))+lagbinsize,
                                    lagbinsize)
                ax.hist(self.flatchain[:,i], bins=lagbins)
            else:
                ax.hist(self.flatchain[:,i], bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        # plt.get_current_fig_manager().toolbar.zoom()
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def break_chain(self, llag_segments):
        """ Break the chain.

        Parameters
        ----------
        llag_segments: list of lists
            list of length 1, wich the single element a two-element array
            bracketing the range of lags (usually the single most probable peak)
            you want to consider for each line.

        """
        if (len(llag_segments) != self.nlc-1):
            print("Error: llag_segments has to be a list of length %d" %
                  (self.nlc-1))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        llag_seq = llag_segments[0]
        if llag_seq is None:
            print("Warning: no rule to break chains with")
        else:
            indx = np.argsort(self.flatchain[:, 2])
            imin, imax = np.searchsorted(self.flatchain[indx, 2], llag_seq)
            indx_cut = indx[imin: imax]
            self.flatchain = self.flatchain[indx_cut,:]
            if hasattr(self, "logp"):
                self.logp = self.logp[indx_cut]

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
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.genfromtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

    def do_pred(self, p_bst=None, fpred=None, dense=10, set_overwrite=True,
                set_decompose=False):
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
        qlist = lnpostfn_photo_p(p_bst, self.zydata, set_retq=True,
                                 set_verbose=False)[4]
        sigma, tau, lags, wids, scales = unpackphotopar(p_bst, self.zydata.nlc,
                                                        hascontlag=True)
        # update qlist
        self.zydata.update_qlist(qlist)
        # initialize PredictRmap object
        P = PredictPmap(zydata=self.zydata, sigma=sigma, tau=tau, lags=lags,
                        wids=wids, scales=scales)
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
        if set_decompose:
            mve_band = (zylclist_pred[0][1] - self.zydata.blist[0])*scales[-1]
            mve_line = (zylclist_pred[1][1] - self.zydata.blist[1])-mve_band
            mve_nonv = jwant * 0.0 + self.zydata.blist[1]
        zydata_pred = LightCurve(zylclist_pred)
        if fpred is not None:
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        if set_decompose:
            return(zydata_pred, [jwant, mve_band, mve_line, mve_nonv])
        else:
            return(zydata_pred)


# ---------------------------------
# SPmap_Model: One-Band Photometric RM

def unpacksbphotopar(p, nlc=1):
    """ Unpack the physical parameters from input 1-d array for single band
    photo mode.
    """
    if nlc != 1:
        raise InputError("SPmap_Model cannot cope with more than one band.")
    sigma = np.exp(p[0])
    tau = np.exp(p[1])
    lag = p[2]
    wid = p[3]
    scale = p[4]
    return(sigma, tau, lag, wid, scale)


def lnpostfn_sbphoto_p(p, zydata, conthpd=None, scalehpd=None,
                       lagtobaseline=0.3, laglimit=None, widtobaseline=0.2,
                       widlimit=None, set_threading=False, blocksize=10000,
                       set_retq=False, set_verbose=False):
    """ log-posterior function of p.

    Parameters
    ----------
    p: array_like
        SPmap_Model parameters, [log(sigma), log(tau), lag1, wid1, scale1]
    zydata: LightCurve object
        Light curve data.
    conthpd: ndarray, optional
        Priors on sigma and tau as an ndarray with shape (3, 2),
        np.array([[log(sigma_low), log(tau_low)],
                  [log(sigma_med), log(tau_med)],
                  [log(sigma_hig), log(tau_hig)]])
        where 'low', 'med', and 'hig' are defined as the 68% confidence
        limits around the median. conthpd usually comes in as an attribute
        of the `Cont_Model` object `hpd` (default: None).
    scalehpd: ndarray, optional
        Prior on ln(scale) as an 1D ndarray with size 3.
        np.array([lnscale_low, lnscale_med, lnscale_hig])
        where 'low', 'med', and 'hig' are defined as the 68% confidence
        limits around the median. Use scalehpd if you have a rough idea of
        how large the ratio of line variation over the underlying continuum is.
    lagtobaseline: float, optional
        Prior on lags. When input lag exceeds lagtobaseline*baseline, a
        logarithmic prior will be applied.
    laglimit: list of tuples.
        hard boundaries for the lag searching.
    widtobaseline: float, optional
        Prior on wids. When input wid exceeds widtobaseline*baseline, a
        logarithmic prior will be applied.
    widlimit: list of tuples, optional
        hard boundaries for the wid searching.
    set_threading: bool, optional
        True if you want threading in filling matrix. It conflicts with the
        'threads' option in Pmap_Model.run_mcmc (default: False).
    blocksize: int, optional
        Maximum matrix block size in threading (default: 10000).
    set_retq: bool, optional
        Return the value(s) of q along with each component of the
        log-likelihood if True (default: False).
    set_verbose: bool, optional
        True if you want verbosity (default: False).
    """
    sigma, tau, lag, wid, scale = unpacksbphotopar(p, zydata.nlc)
    if set_retq:
        vals = list(lnlikefn_sbphoto(zydata, sigma, tau, lag, wid, scale,
                                     set_retq=True, set_verbose=set_verbose,
                                     set_threading=set_threading,
                                     blocksize=blocksize))
    else:
        logl = lnlikefn_sbphoto(zydata, sigma, tau, lag, wid, scale,
                                set_retq=False, set_verbose=set_verbose,
                                set_threading=set_threading,
                                blocksize=blocksize)
    # both conthpd and p[1-2] are in natural log
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
    # for scale
    if scalehpd is not None:
        lnscale = np.log(scale)
        if lnscale < scalehpd[1]:
            prior3 = (lnscale - scalehpd[1])/(scalehpd[1]-scalehpd[0])
        else:
            prior3 = (lnscale - scalehpd[1])/(scalehpd[2]-scalehpd[1])
    else:
        prior3 = 0.0
    # for lags and wids
    prior2 = 0.0
    # penalize on extremely long lags.
    if lagtobaseline < 1.0:
        if np.abs(lag) > lagtobaseline*zydata.rj:
            prior2 += np.log(np.abs(lag)/(lagtobaseline*zydata.rj))
    # penalize long lags to be impossible
    if laglimit is not None:
        if lag > laglimit[0][1] or lag < laglimit[0][0]:
            prior2 += my_pos_inf
    # penalize on extremely large transfer function width
    if widtobaseline < 1.0:
        if np.abs(wid) > lagtobaseline*zydata.rj:
            prior2 += np.log(np.abs(wid)/(lagtobaseline*zydata.rj))
    if widlimit is not None:
        if wid > widlimit[0][1] or wid < widlimit[0][0]:
            prior2 += my_pos_inf
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1+prior3*prior3) - prior2
    if set_retq:
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else:
        logp = logl + prior
        return(logp)


def lnlikefn_sbphoto(zydata, sigma, tau, lag, wid, scale, set_retq=False,
                     set_verbose=False, set_threading=False, blocksize=10000):
    """ Log-likelihood function for the SBmap model.
    """
    if not zydata.issingle:
        raise UsageError("lnlikefn_sbphoto expects a single input light curve.")
    # impossible scenarios
    if (sigma <= 0.0 or tau <= 0.0 or wid < 0.0 or scale < 0.0 or
            lag > zydata.rj):
        return(_exit_with_retval(zydata.nlc, set_retq,
                                 errmsg="Warning: illegal input of parameters",
                                 set_verbose=set_verbose))
    # fill in lags/wids/scales so that we can use spear.py with set_pmap=True.
    lags = np.zeros(3)
    wids = np.zeros(3)
    scales = np.ones(3)
    lags[1] = lag
    wids[1] = wid
    scales[1] = scale
    # we know all elements in zydata.iarr are 1, so we want them to be 2 here.
    if set_threading:
        C = spear_threading(zydata.jarr,zydata.jarr, zydata.iarr+1,
                            zydata.iarr+1,sigma,tau,lags,wids,scales,
                            set_pmap=True, blocksize=blocksize)
    else:
        C = spear(zydata.jarr,zydata.jarr, zydata.iarr+1,zydata.iarr+1,
                  sigma,tau,lags,wids,scales, set_pmap=True)
    # decompose C inplace
    U, info = cholesky(C, nugget=zydata.varr, inplace=True, raiseinfo=False)
    # handle exceptions here
    if info > 0:
        return(_exit_with_retval(
            zydata.nlc, set_retq,
            errmsg="Warning: non positive-definite covariance C #2",
            set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)


class SPmap_Model(object):
    def __init__(self, zydata=None, linename="line"):
        """ SPmap Model object (Single-band Photometric mapping).

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        """
        self.zydata = zydata
        if zydata is None:
            pass
        else:
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            self.names = zydata.names
            # test if all elements in zydata.iarr are one.
            if not np.all(zydata.iarr == 1):
                raise UsageError("Element ids in zydata should all be ones.")
            # number of parameters
            self.ndim = 5
            self.vars = ["sigma", "tau"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            #
            self.vars.append("_".join(["lag",   linename]))
            self.vars.append("_".join(["wid",   linename]))
            self.vars.append("_".join(["scale", linename]))
            self.texs.append("".join([r"$t_{", linename, r"}$"]))
            self.texs.append("".join([r"$w_{", linename, r"}$"]))
            self.texs.append("".join([r"$s_{", linename, r"}$"]))

    def __call__(self, p, **lnpostparams):
        return(lnpostfn_sbphoto_p(p, self.zydata, **lnpostparams))

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """ Do an optimization to find the Maximum a Posterior estimates.

        Parameters
        ----------
        p_ini: array_like
            Pmap_Model parameters [log(sigma), log(tau), lag, wid, scale].

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1, 1, 1, 1] means sigma is fixed while others are
            varying. fixed=[1, 1, 1, 1,] is equivalent to fixed=None
            (default: None).

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
            func = lambda _p: -lnpostfn_sbphoto_p(_p*fixed+p_ini*(1.-fixed),
                                                  self.zydata, **lnpostparams)
        else:
            func = lambda _p: -lnpostfn_sbphoto_p(_p, self.zydata,
                                                  **lnpostparams)
        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, lag, wid, scale = unpacksbphotopar(p_bst,
                                                       nlc=self.zydata.nlc)
        if set_verbose:
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f" % (sigma, tau))
            print("%s %8.3f %s %8.3f %s %8.3f" % (
                self.vars[2], lag,
                self.vars[3], wid,
                self.vars[3], scale,
                ))
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def do_mcmc(self, conthpd=None, scalehpd=None, lagtobaseline=0.3,
                laglimit="baseline", widtobaseline=0.2, widlimit="nyquist",
                nwalkers=100, nburn=100, nchain=100, threads=1, fburn=None,
                fchain=None, flogp=None, set_threading=False, blocksize=10000,
                set_verbose=True):
        """ See `lnpostfn_sbphoto_p` for doc, except for `laglimit` and
        `widlimit`, both of which have different default values
        ('baseline' / 'nyquist').  'baseline' means the boundaries are
        naturally determined by the duration of the light curves,
        and 'nyquist' means the transfer function width has to be within two
        times the typical cadence of light curves.
        """
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
            raise InputError("conflicting set_threading and threads setup:" +
                             "set_threading should be false when threads > 1")
        if laglimit == "baseline":
            laglimit = [[-self.rj, self.rj],]
        elif len(laglimit) != 1:
            raise InputError("laglimit should be a list of a single list")
        if widlimit == "nyquist":
            # two times the cadence, resembling Nyquist sampling.
            widlimit = [[0.0, 2.0*self.cont_cad],]
        elif len(widlimit) != 1:
            raise InputError("widlimit should be a list of a single list")
        # generate array of random numbers
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initialize array
        if conthpd is None:
            p0[:, 0] += np.log(self.cont_std)-0.5
            p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        else:
            # XXX stretch the range from (0,1) to ( conthpd[0,0], conthpd[2,0] )
            p0[:, 0] = p0[:, 0] * (conthpd[2,0] - conthpd[0,0]) + conthpd[0,0]
            p0[:, 1] = p0[:, 1] * (conthpd[2,1] - conthpd[0,1]) + conthpd[0,1]
            # old way, just use 0.5 as the 1\sigma width.
            # p0[:, 0] += conthpd[1,0]-0.5
            # p0[:, 1] += conthpd[1,1]-0.5
        p0[:, 2] = p0[:, 2] * (laglimit[0][1] - laglimit[0][0]) + laglimit[0][0]
        p0[:, 3] = p0[:, 3] * (widlimit[0][1] - widlimit[0][0]) + widlimit[0][0]
        if scalehpd is None:
            pass  # (0, 1) is adequate.
        else:
            # XXX scalehpd is in natural log-space
            p0[:, 4] = np.exp(p0[:, 4] * (scalehpd[2] - scalehpd[0]) +
                              scalehpd[0])
        if set_verbose:
            print("start burn-in")
            if conthpd is None:
                print("no priors on sigma and tau")
            else:
                print("use log-priors on sigma and tau from continuum fitting")
                print(np.exp(conthpd))
            if lagtobaseline < 1.0:
                print("penalize lags longer than %3.2f of the baseline" %
                      lagtobaseline)
            else:
                print("no penalizing long lags, restrict to < laglimit")
            if widtobaseline < 1.0:
                print("penalize wids longer than %3.2f of the baseline" %
                      widtobaseline)
            else:
                print("no penalizing long wids, restrict to < widlimit")
            if scalehpd is None:
                print("no priors on scale")
            else:
                print("using log-priors on scale")
                print(np.exp(scalehpd))
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"
                  % (nburn, nwalkers, nburn*nwalkers))
        # initialize the ensemble sampler
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_sbphoto_p,
                                  args=(self.zydata, conthpd, scalehpd,
                                        lagtobaseline, laglimit, widtobaseline,
                                        widlimit, set_threading, blocksize,
                                        False, False), threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler
        sampler.reset()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        af = sampler.acceptance_fraction
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
                    print("low: %8.3f med %8.3f hig %8.3f" %
                          tuple(hpd[:,i]))
        # register hpd to attr
        self.hpd = hpd

    def get_bfp(self):
        self.bfp = _get_bfp(self.flatchain, self.logp)

    def show_hist(self, bins=100, lagbinsize=1.0, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins for parameters except for 'lag' (default:100).

        lagbinsize: integer, optional
            bin width for 'lag' (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).

        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(14, 2.8*2))
        for i in xrange(2):
            ax = fig.add_subplot(2,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        # line
        for i in xrange(2, 5):
            ax = fig.add_subplot(2,3,i+1+1)
            if np.mod(i, 3) == 2:
                # lag plots
                lagbins = np.arange(int(np.min(self.flatchain[:,i])),
                                    int(np.max(self.flatchain[:,i]))+lagbinsize,
                                    lagbinsize)
                ax.hist(self.flatchain[:,i], bins=lagbins)
            else:
                ax.hist(self.flatchain[:,i], bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def break_chain(self, llag_segments):
        """ Break the chain.

        Parameters
        ----------
        llag_segments: list of lists
            list of a single list, which is a two-element array
            bracketing the range of lags (usually the single most probable
            peak).

        """
        if (len(llag_segments) != 1):
            print("Error: llag_segments has to be a list of length 1")
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        llag_seq = llag_segments[0]
        if llag_seq is None:
            print("Warning: no rule to break chains with")
        else:
            indx = np.argsort(self.flatchain[:, 2])
            imin, imax = np.searchsorted(self.flatchain[indx, 2], llag_seq)
            indx_cut = indx[imin: imax]
            self.flatchain = self.flatchain[indx_cut,:]
            if hasattr(self, "logp"):
                self.logp = self.logp[indx_cut]

    def restore_chain(self):
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
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.genfromtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

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
        qlist = lnpostfn_sbphoto_p(p_bst, self.zydata, set_retq=True,
                                   set_verbose=False)[4]
        sigma, tau, lag, wid, scale = unpacksbphotopar(p_bst, self.zydata.nlc)
        # update qlist
        self.zydata.update_qlist(qlist)
        # initialize PredictRmap object
        P = PredictSPmap(zydata=self.zydata, sigma=sigma, tau=tau, lag=lag,
                         wid=wid, scale=scale)
        nwant = dense*self.cont_npt
        jwant0 = self.jstart - 0.1*self.rj
        jwant1 = self.jend + 0.1*self.rj
        jwant = np.linspace(jwant0, jwant1, nwant)
        zylclist_pred = []
        iwant = np.ones(nwant)
        mve, var = P.mve_var(jwant, iwant)
        sig = np.sqrt(var)
        zylclist_pred.append([jwant, mve, sig])
        zydata_pred = LightCurve(zylclist_pred)
        if fpred is not None:
            zydata_pred.save(fpred, set_overwrite=set_overwrite)
        return(zydata_pred)


# ---------------------------------
# SCmap_Model: Smoothed Continuum Spectroscopic RM

def unpackscspearpar(p, nlc=None):
    """ Unpack the physical parameters from input 1-d array for smoothed
    continuum spec mode.  """
    if nlc is None:
        # possible to figure out nlc from the size of p, only one extra
        # parameter compared to the regular Rmap model.
        nlc = (len(p) - 3)//3 + 1
    sigma = np.exp(p[0])
    tau = np.exp(p[1])
    # XXX have an imaginary unsmoothed light curve, for easily calling spear.
    lags = np.zeros(nlc+1)
    wids = np.zeros(nlc+1)
    scales = np.ones(nlc+1)
    # for the smoothed continuum
    wids[1] = p[2]
    for i in xrange(1, nlc):
        lags[i+1] = p[3+(i-1)*3]
        wids[i+1] = p[4+(i-1)*3]
        scales[i+1] = p[5+(i-1)*3]
    return(sigma, tau, lags, wids, scales)


def lnpostfn_scspear_p(p, zydata, lagtobaseline=0.3, laglimit=None,
                       set_threading=False, blocksize=10000, set_retq=False,
                       set_verbose=False):
    """ log-posterior function of p.

    Parameters
    ----------
    p: array_like
        SCmap_Model parameters, [log(sigma), log(tau), wid0, lag1, wid1, scale1,
        ...]
    zydata: LightCurve object
        Input LightCurve data.
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

    Returns
    -------
    retval: float (set_retq is False) or list (set_retq is True)
        if `retval` returns a list, then it contains the full posterior info
        as a list of [log_posterior, chi2_component, det_component,
        DC_penalty, correction_to_the_mean].

    """
    # unpack the parameters from p
    sigma, tau, lags, wids, scales = unpackscspearpar(p, zydata.nlc)
    if set_retq:
        vals = list(lnlikefn_scspear(zydata, sigma, tau, lags, wids, scales,
                                     set_retq=True, set_verbose=set_verbose,
                                     set_threading=set_threading,
                                     blocksize=blocksize))
    else:
        logl = lnlikefn_scspear(zydata, sigma, tau, lags, wids, scales,
                                set_retq=False, set_verbose=set_verbose,
                                set_threading=set_threading,
                                blocksize=blocksize)
    # XXX deprecated by left here for conformity.
    prior0 = 0.0
    prior1 = 0.0
    # for each lag
    prior2 = 0.0
    for _i in xrange(zydata.nlc-1):
        i = _i + 2
        if lagtobaseline < 1.0:
            if np.abs(lags[i]) > lagtobaseline*zydata.rj:
                # penalize long lags when they are larger than 0.3 times the
                # baseline,
                # as it is too easy to fit the model with non-overlapping
                # signals in the light curves.
                prior2 += np.log(np.abs(lags[i])/(lagtobaseline*zydata.rj))
        # penalize long lags to be impossible
        if laglimit is not None:
            # laglimit starts with the 1st line lightcurve.
            if lags[i] > laglimit[_i][1] or lags[i] < laglimit[_i][0]:
                # try not stack priors
                prior2 = my_pos_inf
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1) - prior2
    if set_retq:
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else:
        logp = logl + prior
        return(logp)


def lnlikefn_scspear(zydata, sigma, tau, lags, wids, scales, set_retq=False,
                     set_verbose=False, set_threading=False, blocksize=10000):
    """ Internal function to calculate the log likelihood.
    """
    # impossible scenarios
    if (sigma <= 0.0 or tau <= 0.0 or np.min(wids) < 0.0 or
            np.min(scales) <= 0.0 or np.max(np.abs(lags)) > zydata.rj):
        return(_exit_with_retval(zydata.nlc, set_retq,
                                 errmsg="Warning: illegal input of parameters",
                                 set_verbose=set_verbose))
    # calculate covariance matrix
    # here we have to trick the program to think that we have line lightcurves
    # with zero lag rather than having a unsmoothed continuum by increasing
    # iarr by one.
    if set_threading:
        C = spear_threading(zydata.jarr, zydata.jarr, zydata.iarr+1,
                            zydata.iarr+1, sigma, tau, lags, wids, scales,
                            blocksize=blocksize)
    else:
        C = spear(zydata.jarr, zydata.jarr, zydata.iarr+1, zydata.iarr+1,
                  sigma, tau, lags, wids, scales)
    # decompose C inplace
    U, info = cholesky(C, nugget=zydata.varr, inplace=True, raiseinfo=False)
    # handle exceptions here
    if info > 0:
        return(_exit_with_retval(
            zydata.nlc, set_retq,
            errmsg="Warning: non positive-definite covariance C #1",
            set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, set_retq=set_retq,
                            set_verbose=set_verbose)
    return(retval)


class SCmap_Model(object):
    def __init__(self, zydata=None):
        """ SCmap Model object.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Light curve data.

        """
        self.zydata = zydata
        if zydata is None:
            pass
        else:
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.cont_std = zydata.cont_std
            self.rj = zydata.rj
            self.jstart = zydata.jstart
            self.jend = zydata.jend
            self.names = zydata.names
            # number of parameters
            self.ndim = 2 + (self.nlc-1)*3 + 1
            self.vars = ["sigma", "tau", "smoothing"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$", r"$t_c$"]
            for i in xrange(1, self.nlc):
                self.vars.append("_".join(["lag", self.names[i]]))
                self.vars.append("_".join(["wid", self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append("".join([
                    r"$t_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))
                self.texs.append("".join([
                    r"$w_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))
                self.texs.append("".join([
                    r"$s_{", self.names[i].lstrip(r"$").rstrip(r"$"), r"}$"]))

    def __call__(self, p, **lnpostparams):
        """ Calculate the posterior value given one parameter set `p`.
        See `lnpostfn_spear_p` for doc.

        Parameters
        ----------
        p: array_like
            Rmap_Model parameters, [log(sigma), log(tau), lag1, wid1,
            scale1, ...]

        lnpostparams: kwargs
            Keyword arguments for `lnpostfn_scspear_p`.

        Returns
        -------
        retval: float (set_retq is False) or list (set_retq is True)
            if `retval` returns a list, then it contains the full posterior info
            as a list of [log_posterior, chi2_component, det_component,
            DC_penalty, correction_to_the_mean].

        """
        return(lnpostfn_scspear_p(p, self.zydata, **lnpostparams))

    def do_map(self, p_ini, fixed=None, **lnpostparams):
        """ Do an optimization to find the Maximum a Posterior estimates.
        See `lnpostfn_scspear_p` for doc.

        Parameters
        ----------
        p_ini: array_like
            SCmap_Model parameters, [log(sigma), log(tau), wid0, lag1,
            wid1, scale1, ...]

        fixed: array_like, optional
            Same dimension as p_ini, but with 0 for parameters that is fixed in
            the optimization, and with 1 for parameters that is varying, e.g.,
            fixed = [0, 1, 1, 1, 1, 1, ...] means sigma is fixed while
            others are varying. fixed=[1, 1, 1, 1, 1, 1, ...] is
            equivalent to fixed=None (default: None).

        lnpostparams: kwargs
            Kewword arguments for `lnpostfn_scspear_p`.

        Returns
        -------
        p_bst: array_like
            Best-fit parameters.

        l: float
            The maximum log-posterior.

        """
        set_verbose = lnpostparams.pop("set_verbose", True)
        set_retq = lnpostparams.pop("set_retq",    False)
        if set_retq is True:
            raise InputError("set_retq has to be False")
        p_ini = np.asarray(p_ini)
        if fixed is not None:
            fixed = np.asarray(fixed)
            func = lambda _p: -lnpostfn_scspear_p(_p*fixed+p_ini*(1.-fixed),
                                                  self.zydata, **lnpostparams)
        else:
            func = lambda _p: -lnpostfn_scspear_p(
                _p, self.zydata, **lnpostparams)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if fixed is not None:
            p_bst = p_bst*fixed+p_ini*(1.-fixed)
        sigma, tau, lags, wids, scales = unpackscspearpar(p_bst,
                                                          self.zydata.nlc)
        if set_verbose:
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f wid0 %8.3f " % (sigma, tau, wids[0]))
            for i in xrange(self.nlc-1):
                ip = 2+i*3 + 1
                print("%s %8.3f %s %8.3f %s %8.3f" % (
                    self.vars[ip+0], lags[i+1],
                    self.vars[ip+1], wids[i+1],
                    self.vars[ip+2], scales[i+1],
                    ))
            print("with logp  %10.5g " % -v_bst)
        return(p_bst, -v_bst)

    def do_mcmc(self, lagtobaseline=0.3, laglimit="baseline", nwalkers=100,
                nburn=100, nchain=100, threads=1, fburn=None, fchain=None,
                flogp=None, set_threading=False, blocksize=10000,
                set_verbose=True):
        """ Run MCMC sampling over the parameter space.

        Parameters
        ----------
        lagtobaseline: float, optional
            Prior on lags. When input lag exceeds lagtobaseline*baseline, a
            logarithmic prior will be applied.
        laglimit: str or list of tuples.
            Hard boundaries for the lag searching during MCMC sampling.
            'baseline' means the boundaries are naturally determined by
            the duration of the light curves, or you can set them as a list
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
        """
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
            raise InputError("laglimit should be a list of nline lists")
        # generate array of random numbers
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        # initialize array
        # sigma and tau without prior
        p0[:, 0] += np.log(self.cont_std)-0.5
        p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        # make the initial wid0 to be [0, 10*cadence]
        p0[:, 2] *= 10. * self.cont_cad
        for i in xrange(self.nlc-1):
            p0[:, 3+i*3] = p0[:,3+i*3] * (laglimit[i][1] -
                                          laglimit[i][0]) + laglimit[i][0]
        if set_verbose:
            print("start burn-in")
            if lagtobaseline < 1.0:
                print("penalize lags longer than %3.2f of the baseline" %
                      lagtobaseline)
            else:
                print("no penalizing long lags, restrict to < baseline")
            print("nburn: %d nwalkers: %d --> number of burn-in iterations: %d"
                  % (nburn, nwalkers, nburn*nwalkers))
        # initialize the ensemble sampler
        sampler = EnsembleSampler(nwalkers, self.ndim, lnpostfn_scspear_p,
                                  args=(self.zydata, lagtobaseline, laglimit,
                                        set_threading, blocksize, False, False),
                                  threads=threads)
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        if set_verbose:
            print("burn-in finished")
        if fburn is not None:
            if set_verbose:
                print("save burn-in chains to %s" % fburn)
            np.savetxt(fburn, sampler.flatchain)
        # reset the sampler
        sampler.reset()
        if set_verbose:
            print("start sampling")
        sampler.run_mcmc(pos, nchain, rstate0=state)
        if set_verbose:
            print("sampling finished")
        af = sampler.acceptance_fraction
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
                    print("low: %8.3f med %8.3f hig %8.3f" %
                          tuple(hpd[:,i]))
        # register hpd to attr
        self.hpd = hpd

    def get_bfp(self):
        self.bfp = _get_bfp(self.flatchain, self.logp)

    def show_hist(self, bins=100, lagbinsize=1.0, figout=None, figext=None):
        """ Display histograms of the posterior distributions.

        Parameters
        ----------
        bins: integer, optional
            Number of bins for parameters except for 'lag' (default:100).

        lagbinsize: integer, optional
            bin width for 'lag' (default:100).

        figout: str, optional
            Output figure name (default: None, i.e., using sequencial integers).

        figext: str, optional
            Output figure extension (default: None, i.e., using `show`).

        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig = plt.figure(figsize=(14, 2.8*self.nlc))
        for i in xrange(2):
            ax = fig.add_subplot(self.nlc,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, bins)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        # for wid0
        ax = fig.add_subplot(self.nlc,3,3)
        ax.hist(self.flatchain[:,2], bins)
        ax.set_xlabel(self.texs[2])
        ax.set_ylabel("N")
        # go to lines
        for k in xrange(self.nlc-1):
            for i in xrange(3+k*3, 6+k*3):
                ax = fig.add_subplot(self.nlc,3,i+1)
                if np.mod(i, 3) == 0:
                    # lag plots
                    lagbins = np.arange(int(np.min(self.flatchain[:,i])),
                                        int(np.max(self.flatchain[:,i])) +
                                        lagbinsize, lagbinsize)
                    ax.hist(self.flatchain[:,i], bins=lagbins)
                else:
                    ax.hist(self.flatchain[:,i], bins)
                ax.set_xlabel(self.texs[i])
                ax.set_ylabel("N")
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
        if (len(llag_segments) != self.nlc-1):
            print("Error: llag_segments has to be a list of length %d" %
                  (self.nlc-1))
            return(1)
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        for i, llag_seq in enumerate(llag_segments):
            if llag_seq is None:
                continue
            indx = np.argsort(self.flatchain[:, 3+i*3])
            imin, imax = np.searchsorted(self.flatchain[indx, 3+i*3], llag_seq)
            indx_cut = indx[imin: imax]
            self.flatchain = self.flatchain[indx_cut,:]
            if hasattr(self, "logp"):
                self.logp = self.logp[indx_cut]

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
        self.flatchain = np.genfromtxt(fchain)
        self.flatchain_whole = np.copy(self.flatchain)
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)
        if flogp is not None:
            self.logp = np.genfromtxt(flogp)
            self.logp_whole = np.copy(self.logp)
            self.get_bfp()

    def get_qlist(self, p_bst):
        """ get the best-fit linear responses.

        Parameters
        ----------
        p_bst: list
            best-fit parameters.
        """
        self.qlist = lnpostfn_scspear_p(p_bst, self.zydata, set_retq=True,
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
        sigma, tau, lags, wids, scales = unpackscspearpar(p_bst,
                                                          self.zydata.nlc)
        # update qlist
        self.zydata.update_qlist(self.qlist)
        # initialize PredictRmap object
        P = PredictSCmap(zydata=self.zydata, sigma=sigma, tau=tau,
                         lags=lags, wids=wids, scales=scales)
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
