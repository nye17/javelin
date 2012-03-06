#Last-modified: 05 Mar 2012 08:00:09 PM

from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
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

my_neg_inf = float(-1.0e+300)

def unpackspearpar(p) :
    """ Unpack the physical parameters from input array.
    """
    sigma = np.exp(p[0])
    tau   = np.exp(p[1])
    llags   = np.zeros(zydata.nlc-1)
    lwids   = np.zeros(zydata.nlc-1)
    lscales =  np.ones(zydata.nlc-1)
    for i in xrange(zydata.nlc-1) : 
        llags[i]   = p[2+i*3]
        lwids[i]   = p[3+i*3]
        lscales[i] = p[4+i*3]
    return(sigma, tau, llags, lwids, lscales)

def lnpostfn_spear_p(p, zydata, conthpd=None, lagtobaseline=0.3, 
        set_retq=False, set_verbose=False, 
        set_threading=False, blocksize=10000):
    """ log-posterior function of p.
    """
    # unpack the parameters from p
    sigma, tau, llags, lwids, lscales = unpackspearpar(p)
    if set_retq :
        vals = lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
                retq=True, set_verbose=set_verbose,
                set_threading=set_threading, blocksize=blocksize)
    else :
        logl = lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
                retq=False, set_verbose=set_verbose,
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
    prior3 = 0.0
    if lagtobaseline < 1.0 :
        for i in xrange(zydata.nlc-1) :
            if np.abs(llags[i]) > lagtobaseline*zydata.rj :
                # penalize long lags when they are larger than 0.3 times the baseline,
                # as it is too easy to fit the model with non-overlapping
                # signals in the light curves.
                prior3 += np.log(np.abs(llags[i])/(lagtobaseline*zydata.rj))
    # add logp of all the priors
    prior = -0.5*(prior0*prior0+prior1*prior1) - prior3
    if set_retq :
        vals[0] = vals[0] + prior
        vals.extend([prior0, prior1, prior2])
        return(vals)
    else :
        logp = logl + prior
        return(logp)

def lnlikefn_spear(zydata, sigma, tau, llags, lwids, lscales, 
        retq=False, set_verbose=False, 
        set_threading=False, blocksize=10000):
    if zydata.issingle:
        raise UsageError("lnlikefn_spear does not work for single mode")
    # impossible scenarios
    if (sigma<=0.0 or tau<=0.0 or np.min(lwids)<0.0 or np.min(lscales)<=0.0
                   or np.max(np.abs(llags))>zydata.rj) :
       return(_exit_with_retval(zydata.nlc, retq, 
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
       return(_exit_with_retval(zydata.nlc, retq, 
              errmsg="Warning: non positive-definite covariance C", 
              set_verbose=set_verbose))
    retval = _lnlike_from_U(U, zydata, retq=retq, set_verbose=set_verbose)
    return(retval)


def _lnlike_from_U(U, zydata, retq=False, set_verbose=False):
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
            return(_exit_with_retval(zydata.nlc, retq, 
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
    if retq:
        q = np.dot(d, a)
        return(_log_like, _chi2, _compl_pen, _wmean_pen, q)
    else:
        return(_log_like)

def _exit_with_retval(nlc, retq, errmsg=None, set_verbose=False):
    """ Return failure elegantly.
    
    When you are desperate and just want to leave the calculation with 
    appropriate return values that quietly speak out your angst.
    """
    if errmsg is not None:
        if set_verbose:
            print("Exit: %s"%errmsg)
    if retq:
        return(my_neg_inf, my_neg_inf, my_neg_inf, my_neg_inf, 
              [my_neg_inf]*nlc)
    else:
        return(my_neg_inf)


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
            self.nlc = zydata.nlc
            self.npt = zydata.npt
            self.cont_npt = zydata.nptlist[0]
            self.cont_cad = zydata.cont_cad
            self.rj  = zydata.rj
            self.jstart = zydata.jstart
            self.jend   = zydata.jend
            # number of parameters
            self.ndim = 2 + (self.nlc-1)*3
            self.names = zydata.names
            self.vars = ["sigma", "tau"]
            self.texs = [r"$\log\,\sigma$", r"$\log\,\tau$"]
            for i in xrange(1, self.nlc) :
                self.vars.append("_".join(["lag",   self.names[i]]))
                self.vars.append("_".join(["wid",   self.names[i]]))
                self.vars.append("_".join(["scale", self.names[i]]))
                self.texs.append( "".join([r"$t_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$w_{", self.names[i] ,r"}$"]))
                self.texs.append( "".join([r"$s_{", self.names[i] ,r"}$"]))

    def do_map(self, p_ini, fixed=None, conthpd=None, set_verbose=True):
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

        conthpd: ndarray, optional
            Priors on sigma and tau as an ndarray with shape (3, 2), 
            np.array([[log(sigma_low), log(tau_low)],
                      [log(sigma_med), log(tau_med)],
                      [log(sigma_hig), log(tau_hig)]])
            where 'low', 'med', and 'hig' are defined as the 68% confidence
            limits around the median. conthpd usually comes in as an attribute
            of the DRW_Model object DRW_Model.hpd (default: None).

        set_verbose: bool, optional
            True if best-fit parameters are printed to std (default: True).

        Returns
        -------
        p_bst : array_like
            Best-fit parameters.

        l: float
            The maximum log-posterior.

        """

        p_ini = np.asarray(p_ini)
        if fixed is not None :
            fixed = np.asarray(fixed)
            func = lambda _p : -self.__call__(_p*fixed+p_ini*(1.-fixed),
                    conthpd=conthpd)
        else :
            func = lambda _p : -self.__call__(_p, conthpd=conthpd)

        p_bst, v_bst = fmin(func, p_ini, full_output=True)[:2]
        if set_verbose :
            print("Best-fit parameters are")
            print("sigma %8.3f tau %8.3f"%tuple(np.exp(p_bst[0:2])))
            for i in xrange(1,self.nlc) :
                ip = 2+(i-1)*3
                print("%s %8.3f %s %8.3f %s %8.3f"%(
                    self.vars[ip+0], p_bst[ip+0],
                    self.vars[ip+1], p_bst[ip+1],
                    self.vars[ip+2], p_bst[ip+2],
                    ))
            print("with logp  %10.5g "%-v_bst)
        return(p_bst, -v_bst)


    def do_mcmc(self, conthpd=None, lagtobaseline=0.3, 
            nwalkers=100, nburn=100, nchain=100,
            fburn=None, fchain=None, set_verbose=True):
        """ test
        """
        p0 = np.random.rand(nwalkers*self.ndim).reshape(nwalkers, self.ndim)
        if conthpd is None:
            p0[:, 0] += np.log(self.cont_std)-0.5
            p0[:, 1] += np.log(np.sqrt(self.rj*self.cont_cad))-0.5
        else :
            p0[:, 0] += conthpd[1,0]-0.5
            p0[:, 1] += conthpd[1,1]-0.5
        for i in xrange(self.nlc-1) :
            p0[:, 2+i*3] = p0[:,2+i*3]*self.rj*lagtobaseline
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
                args=(self.zydata, conthpd, lagtobaseline,), threads=2)
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
        # make chain an attritue
        self.flatchain = sampler.flatchain
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

    def show_hist(self):
        """ Plot the histograms.
        """
        if not hasattr(self, "flatchain"):
            print("Warning: need to run do_mcmc or load_chain first")
            return(1)
        ln10 = np.log(10.0)
        fig  = plt.figure(figsize=(6*3, 3*self.nlc))
        for i in xrange(2) :
            ax = fig.add_subplot(self.nlc,3,i+1)
            ax.hist(self.flatchain[:,i]/ln10, 100)
            ax.set_xlabel(self.texs[i])
            ax.set_ylabel("N")
        for k in xrange(self.nlc-1):
            for i in xrange(2+k*3, 5+k*3) :
                ax = fig.add_subplot(self.nlc,3,i+1+1) 
                ax.hist(self.flatchain[:,i], 100)
                ax.set_xlabel(self.texs[i])
                ax.set_ylabel("N")
        plt.show()


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
        self.ndim = self.flatchain.shape[1]
        # get HPD
        self.get_hpd(set_verbose=set_verbose)

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

        sigma = np.exp(p_bst[0])
        tau   = np.exp(p_bst[1])
        llags   = np.zeros(self.nlc-1)
        lwids   = np.zeros(self.nlc-1)
        lscales =  np.ones(self.nlc-1)
        lags    = np.zeros(self.nlc)
        wids    = np.zeros(self.nlc)
        scales  =  np.ones(self.nlc)
        for i in xrange(self.nlc-1) : 
            llags[i]   = p_bst[2+i*3]
            lwids[i]   = p_bst[3+i*3]
            lscales[i] = p_bst[4+i*3]
            lags[i+1]  = p_bst[2+i*3]
            wids[i+1]  = p_bst[3+i*3]
            scales[i+1]= p_bst[4+i*3]
        qlist  = lnlikefn_spear(self.zydata, sigma, tau, llags, lwids, lscales, 
                retq=True)[4]
        self.zydata.update_qlist(qlist)
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

#def _test(self, p):
#    return(np.sum(p))



if __name__ == "__main__":    
    import matplotlib.pyplot as plt

    sigma, tau = (2.00, 100.0)
    lagy, widy, scaley = (150.0,  3.0, 2.0)
    lagz, widz, scalez = (200.0,  9.0, 0.5)
    lags   = np.array([0.0,   lagy,   lagz])
    wids   = np.array([0.0,   widy,   widz])
    scales = np.array([1.0, scaley, scalez])

    if False :
        lcfile = "dat/loopdeloop_con.dat"
        zydata   = get_data(lcfile)
        cont   = DRW_Model(zydata)
#        cont.do_mcmc(nwalkers=100, nburn=50, nchain=50, fburn="burn0.dat",
#                fchain="chain0.dat")
#
#        p_ini = [0.0, 1.0]
#        cont.do_map(p_ini, fixed=None, set_prior=False, rank="Full", 
#            set_verbose=True)
#
#        cont = DRW_Model()
        cont.load_chain("chain0.dat")
        cont.show_hist()
#        p_bst = [cont.hpd[1, 0], cont.hpd[1,1]]
#        zypred = cont.do_pred(p_bst, fpred="dat/loopdeloop_con.p.dat", dense=10)
#        zypred.plot(set_pred=True, obs=zydata)

    if True :
        from lcmodel import DRW_Model
        lcfile = "dat/loopdeloop_con_y.dat"
        zydata   = get_data(lcfile)
        rmap   = Rmap_Model(zydata)

        cont   = DRW_Model()
        cont.load_chain("dat/chain0.dat")

        rmap.do_mcmc(zydata, cont.hpd, nwalkers=100, nburn=50,
                nchain=50, fburn=None, fchain="dat/test.dat")

#        rmap.load_chain("chain6.dat")
#        rmap.load_chain("chain5.dat")
#        rmap.show_hist()
#        rmap.break_chain([[0, 400],])
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
        cont.load_chain("chain0.dat")

#        rmap.do_mcmc(cont.hpd, nwalkers=100, nburn=50,
#                nchain=50, fburn="burndou1.dat", fchain="chaindou1.dat")

        rmap.load_chain("chaindou1.dat")
        rmap.show_hist()
        rmap.break_chain([[0,400],[0,400]])
        rmap.show_hist()
        rmap.get_hpd()

    if False :
        lcfile = "dat/Arp151/Arp151_B.dat"
        zydata   = get_data(lcfile)
        cont   = DRW_Model(zydata)
#        cont.do_mcmc(nwalkers=100, nburn=50, nchain=50, fburn=None,
#                fchain="chain_arp151_B.dat")
#        cont.load_chain("chain_arp151_B.dat")
#        cont.show_hist()

    if False :
        lcfile = "dat/Arp151/Arp151_B.dat"
        zydata   = get_data(lcfile)
        cont   = DRW_Model(zydata)
        cont.load_chain("chain_arp151_B.dat")
        cont.get_hpd()

        lcfiles = ["dat/Arp151/Arp151_B.dat",
                   "dat/Arp151/Arp151_V.dat",
                   "dat/Arp151/Arp151_Halpha.dat",
                   "dat/Arp151/Arp151_Hbeta.dat",
                   "dat/Arp151/Arp151_Hgamma.dat",
                   ]
        zydata   = get_data(lcfiles)
        rmap   = Rmap_Model(zydata)
#        rmap.do_mcmc(cont.hpd, nwalkers=100, nburn=50,
#                nchain=50, fburn="burn_arp151_2.dat", fchain="chain_arp151_2.dat")

#        rmap.load_chain("chain_arp151_2.dat")
        rmap.load_chain("chain_arp151_5.dat")
#        rmap.show_hist()
#        rmap.get_hpd()
#        rmap.break_chain([[-10,10],])
        rmap.break_chain([[-30,30], None, None, None])
#        rmap.show_hist()
        rmap.get_hpd()

        p_ini = rmap.hpd[1,:]
        p_bst = rmap.do_map(p_ini, fixed=None, conthpd=cont.hpd,
                set_verbose=True)[0]

        zypred = rmap.do_pred(p_bst, fpred="dat/Arp151_5pred.dat", dense=10)
        zypred.plot(set_pred=True, obs=zydata)
        


    exit()
