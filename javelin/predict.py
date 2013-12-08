from gp import Mean, Covariance, observe, Realization, GPutils
from gp import NearlyFullRankCovariance, FullRankCovariance
from cholesky_utils import cholesky, cholesky2, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from spear import spear, spear_threading
from zylc import LightCurve

np.set_printoptions(precision=3)

__all__ = ["PredictSignal", ]

""" Generate random realizations based on the covariance function.
"""

def generateError(e, errcov=0.0):
    """ Generate Gaussian random errors given dispersions and correlation coefficient.

    Parameters
    ----------
    e: array_like
        Diagonal errors.

    errcov: scalar, optional
        Correlation coefficient of errors (default: 0)

    """
    if np.isscalar(e) :
        print("Warning: e is a scalar, proceed as 1-element array")
        e = np.atleast_1d(e)
    nwant = len(e)
    ediag = np.diag(e*e)
    if errcov == 0.0 :
        ecovmat = ediag
    else :
        temp1 = np.repeat(e, nwant).reshape(nwant,nwant)
        temp2 = (temp1*temp1.T - ediag)*errcov
        ecovmat = ediag + temp2
    et = multivariate_normal(np.zeros_like(e), ecovmat)
    return(et)

def generateLine(jc, mc, lag, wid, scale, mc_mean=0.0, ml_mean=0.0):
    """ Convolve with the top-hat kernel to get line light curve, however, the
    input continuum signal has to be dense enough and regularly sampled.

    .. note:: this should be deprecated.

    Parameters
    ----------
    jc: array_like
        Continuum epochs, must be regularly sampled.

    mc: array_like
        Continuum signal.

    lag: scalar
        Time lag.

    wid: scalar
        Width of the transfer function.

    scale: scalar
        Ratio of line signal and continuum signal.

    mc_mean: scalar
        Subtracted by mc to get the truth signal (default: 0).

    ml_mean: scalar
        Added to the line signal to get observed line light curve signal
        (.default: 0)

    Returns
    -------

    jl: array_like
        Epochs of line light curve, displaced version of 'jc'.

    ml: array_like
        Line light curve.
        
    """
    djc = jc[1:] - jc[:-1]
    if (np.abs(np.min(djc) - np.max(djc))>0.01) :
        raise RuntimeError("input jc has to be regularly sampled")
    # scale time unit to djc
    junit = djc[0]
    window_len = np.floor(0.5*wid/junit)
    print(window_len)
    # continuum signal 
    sc = mc - mc_mean
    if wid < 0.0 :
        print("WARNING: negative wid? set to abs(wid) %10.4f"%wid)
        wid = np.abs(wid)
    if scale < 0.0 :
        print("WARNING: negative scale? set to abs(scale) %10.4f"%scale)
        scale = np.abs(scale)
    sl = smooth(sc,window_len=window_len,window='flat')
    ml = ml_mean + sl*scale
    jl = jc + lag
    print(len(ml))
    return(jl, ml)

class PredictSignal(object):
    """
    Predict continuum light curves.
    """
    def __init__(self, zydata=None, lcmean=0.0, covfunc="drw", rank="Full", **covparams):
        """ PredictSignal object for simulating continuum light curves.

        Parameters
        ----------
        zydata: LightCurve object, optional
            Observed light curve in LightCurve format, set to 'None' if no observation
            is done (default: Done)

        lcmean: scalar or a Mean object, optional
            Mean amplitude of the underlying signal (default: 0).

        covfunc: string, optional
            Name of the covariance function (default: drw).

        rank: string, optional
            Rank of the covariance function, could potentially use 'NearlyFull'
            rank covariance when the off-diagonal terms become strong (default:
            'Full').

        covparams: kwargs
            Parameters for 'covfunc'.

        """
        # make the Mean object
        try :
            const = float(lcmean)
            meanfunc = lambda x: const*(x*0.0+1.0)
            self.M = Mean(meanfunc)
        except ValueError:
            if isinstance(lcmean, Mean):
                self.M = lcmean
            else:
                raise RuntimeError("lcmean is neither a Mean obj or a const")
        # generate covariance parameters
        covfunc_dict = get_covfunc_dict(covfunc, **covparams)
        if rank is "Full" :
            self.C  = FullRankCovariance(**covfunc_dict)
        elif (rank is "NearlyFull") or rank is ("NearlyFullRankCovariance") :
            self.C  = NearlyFullRankCovariance(**covfunc_dict)
        # observe zydata
        if zydata is None :
            print("No *zydata* Observed, Unconstrained Realization")
        else:
            print("Observed *zydata*, Constrained Realization")
            jdata = zydata.jarr
            mdata = zydata.marr + zydata.blist[0]
            edata = zydata.earr
            observe(self.M, self.C, obs_mesh=jdata, obs_V = edata, obs_vals = mdata)

    def mve_var(self, jwant):
        """ Generate the minimum variance estimate and its associated variance.

        Parameters
        ----------
        jwant: array_like
            Desired epochs for simulated light curve.

        Returns
        -------
        m: array_like
            Minimum variance estimate of the underlying signal.

        v: array_like
            Variance at simulated point.

        """
        m, v = GPutils.point_eval(self.M, self.C, jwant)
        return(m,v)

    def generate(self, jwant, ewant=0.0, num=1, errcov=0.0):
        """ Draw random realizations as simulated light curves.

        Parameters
        ----------
        jwant: array_like
            Desired epochs for simulated light curve.

        ewant: scalar or array_like, optional
            Errors in the simulated light curve (default: 0.0).

        errcov: scalar, optional
            Correlation coefficient of errors (default: 0.0).

        num: scalar, optional
            Number of simulated light curves to be generated.
        
        Returns
        -------
        mwant: array_like (num=1) or list of arrays (num>1)
            Simulated light curve(s)

        """
        if (np.min(ewant) < 0.0):
            raise RuntimeError("ewant should be either 0  or postive")
        elif np.alltrue(ewant==0.0):
            set_error_on_mocklc = False
        else:
            set_error_on_mocklc = True

        # number of desired epochs
        nwant = len(jwant)

        if np.isscalar(ewant):
            e = np.zeros(nwant) + ewant
        else :
            e = ewant

        # generate covariance function
        ediag = np.diag(e*e)
        if errcov == 0.0 :
            ecovmat = ediag
        else :
            temp1 = np.repeat(e, nwant).reshape(nwant,nwant)
            temp2 = (temp1*temp1.T - ediag)*errcov
            ecovmat = ediag + temp2

        if num == 1:
            f = Realization(self.M, self.C)
            mwant = f(jwant)
            if set_error_on_mocklc:
                mwant += multivariate_normal(np.zeros(nwant), ecovmat)
            return(mwant)
        else:
            mwant_list = []
            for i in xrange(num):
                f = Realization(self.M, self.C)
                mwant = f(jwant)
                if set_error_on_mocklc:
                    mwant += multivariate_normal(np.zeros(nwant), ecovmat)
                mwant_list.append(mwant)
            return(mwant_list)

class PredictRmap(object):
    """ Predict light curves for Rmap, with data constraints.
    """
    def __init__(self, zydata, set_threading=False,  **covparams):
        """ PredictRmap object.

        Parameters
        ----------
        zydata: LightCurve object
            Observed light curve in LightCurve format.

        covparams: kwargs
            Parameters for the spear covariance function.

        """
        self.zydata = zydata
        self.covparams = covparams
        self.jd = self.zydata.jarr
        # has to be the true mean instead of the sample mean
        self.md = self.zydata.marr
        self.id = self.zydata.iarr
        self.blist = self.zydata.blist
        # variance
        self.vd = np.power(self.zydata.earr, 2.)
        # preparation
        self.set_threading = set_threading
        self._get_covmat(set_threading=set_threading)
        self._get_cholesky()
        self._get_cplusninvdoty()

    def mve_var(self, jwant, iwant):
        """ Generate the minimum variance estimate and its associated variance.

        Parameters
        ----------
        jwant: array_like
            Desired epochs for simulated light curve.

        iwant: array_like
            Desired ids for simulated light curve.

        Returns
        -------
        m: array_like
            Minimum variance estimate of the underlying signal.

        v: array_like
            Variance at simulated point.
        """
        m, v = self._fastpredict(jwant, iwant, set_threading=self.set_threading)
        for i in xrange(len(jwant)) :
            m[i] += self.blist[int(iwant[i])-1]
        return(m, v)

    def generate(self, zylclist) :
        """ Presumably zylclist has our input j, e, and i, and the values in m
        should be the mean.

        Parameters
        ----------
        zylclist: list of 3-list light curves 
            Pre-simulated light curves in zylclist, with the values in m-column as
            the light curve mean, and those in e-column as the designated errorbar.

        Returns
        -------
        zylclist_new: list of 3-list light curves
            Simulated light curves in zylclist.

        """
        nlc = len(zylclist)
        jlist = []
        mlist = []
        elist = []
        ilist = []
        for ilc, lclist in enumerate(zylclist):
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                if (np.min(msubarr) != np.max(msubarr)) : 
                    print("WARNING: input zylclist has inequal m elements in "+
                          "light curve %d, please make sure the m elements "+
                          "are filled with the desired mean of the mock "+
                          "light curves, now reset to zero"%ilc)
                    msubarr = msubarr * 0.0
                nptlc = len(jsubarr)
                # sort the date, safety
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
                ilist.append(np.zeros(nptlc, dtype="int")+ilc+1)
        zylclist_new = []
        for ilc in xrange(nlc) :
            m, v = self.mve_var(jlist[ilc], ilist[ilc])
            # no covariance considered here
            vcovmat = np.diag(v)
            if (np.min(elist[ilc]) < 0.0):
                raise RuntimeError("error for light curve %d should be either 0 or postive"%ilc)
            elif np.alltrue(elist[ilc]==0.0):
                set_error_on_mocklc = False
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) 
            else:
                set_error_on_mocklc = True
                ecovmat = np.diag(elist[ilc]*elist[ilc])
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) + multivariate_normal(np.zeros_like(m), ecovmat)
            zylclist_new.append([jlist[ilc], mlist[ilc], elist[ilc]])
        return(zylclist_new)

    def _get_covmat(self, set_threading=False) :
        if set_threading :
            self.cmatrix = spear_threading(self.jd,self.jd,self.id,self.id, **self.covparams)
        else :
            self.cmatrix = spear(self.jd,self.jd,self.id,self.id, **self.covparams)
        print("covariance matrix calculated")

    def _get_cholesky(self) :
        self.U = cholesky(self.cmatrix, nugget=self.vd, inplace=True, raiseinfo=True)
        print("covariance matrix decomposed and updated by U")

    def _get_cplusninvdoty(self) :
        # now we want cpnmatrix^(-1)*mag = x, which is the same as
        #    mag = cpnmatrix*x, so we solve this equation for x
        self.cplusninvdoty = chosolve_from_tri(self.U, self.md, nugget=None, inplace=False)

    def _fastpredict(self, jw, iw, set_threading=False) :
        """ jw : jwant
            iw : iwant
        """
        mw = np.zeros_like(jw)
        vw = np.zeros_like(jw)
        for i, (jwant, iwant) in enumerate(zip(jw, iw)):
            if set_threading :
                covar = spear_threading(jwant,self.jd,iwant,self.id, **self.covparams)
            else :
                covar = spear(jwant,self.jd,iwant,self.id, **self.covparams)
            cplusninvdotcovar = chosolve_from_tri(self.U, covar.T, nugget=None, inplace=False)
            if set_threading :
                vw[i] = spear_threading(jwant, jwant, iwant, iwant, **self.covparams)
            else :
                vw[i] = spear(jwant, jwant, iwant, iwant, **self.covparams)
            mw[i] = np.dot(covar, self.cplusninvdoty)
            vw[i] = vw[i] - np.dot(covar, cplusninvdotcovar)
        return(mw, vw)

class PredictSpear(object):
    """ Generate continuum and line light curves for 'Rmap', 'Pmap', and 'SPmap' `spearmodes`, but without data constraint.
    """
    def __init__(self, sigma, tau, llags, lwids, lscales, spearmode="Rmap"):
        """

        llags, lwids, lscales: properties of the line transfer functions, all lists of length n_line.

        1) if spearmode is "Rmap" , the transfer functions have nline elmements.
        2) if spearmode is "Pmap" , the transfer functions have     2 elmements, first for the line, second for the continuum under line band with lag=0 and width=0.
        3) if spearmode is "SPmap", the transfer functions have     2 elmement, first for the line, second with lag=0 and width=0.
        """
        self.sigma = sigma
        self.tau   = tau
        # number of light curves including continuum
        self.nlc   = len(llags) + 1
        self.lags  = np.zeros(self.nlc)
        self.wids  = np.zeros(self.nlc)
        self.scales = np.ones(self.nlc)
        self.lags[1 :]   = llags
        self.wids[1 :]   = lwids
        self.scales[1 :] = lscales
        # mode, sanity checks on array sizes.
        self.spearmode = spearmode
        if self.spearmode == "Rmap" :
            self.nlc_obs = self.nlc
        elif self.spearmode == "Pmap" :
            self.nlc_obs = 2
            if self.nlc != 3 :
                raise RuntimeError("Pmap mode expects 2 elements in each transfer function array")
        elif self.spearmode == "SPmap" :
            self.nlc_obs = 1
            if self.nlc != 3 :
                raise RuntimeError("SPmap mode expects 2 elements in each transfer function array")

    def generate(self, zylclist, set_threading=False) :
        """ Presumably zylclist has our input j, and e, and the values in m
        should be the mean.

        Parameters
        ----------
        zylclist: list of 3-list light curves 
            Pre-simulated light curves in zylclist, with the values in m-column as
            the light curve mean, and those in e-column as the designated errorbar.

        Returns
        -------
        zylclist_new: list of 3-list light curves
            Simulated light curves in zylclist.

        """
        nlc_obs = len(zylclist)
        if nlc_obs != self.nlc_obs :
            raise RuntimeError("zylclist has unmatched nlc_obs with spearmode %s" % self.spearmode)
        jlist = []
        mlist = []
        elist = []
        ilist = []
        nptlist = []
        npt   = 0
        for ilc, lclist in enumerate(zylclist):
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                if (np.min(msubarr) != np.max(msubarr)) : 
                    print("WARNING: input zylclist has inequal m elements in light curve %d," +
                           "please make sure the m elements are filled with the desired mean" +
                           "of the mock light curves, now reset to zero"%ilc)
                    msubarr = msubarr * 0.0
                nptlc = len(jsubarr)
                # sort the date, safety
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
                ilist.append(np.zeros(nptlc, dtype="int")+ilc+1) # again, starting at 1.
                npt += nptlc
                nptlist.append(nptlc)
        # collapse the list to one array
        jarr, marr, earr, iarr = self._combineddataarr(npt, nptlist, jlist, mlist, elist, ilist)
        # get covariance function
        if set_threading :
            if self.spearmode == "Rmap" :
                cmatrix = spear_threading(jarr, jarr, iarr  , iarr  , self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=False)
            elif self.spearmode == "Pmap" :
                cmatrix = spear_threading(jarr, jarr, iarr  , iarr  , self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=True)
            elif self.spearmode == "SPmap" :
                cmatrix = spear_threading(jarr, jarr, iarr+1, iarr+1, self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=True) 
        else :
            if self.spearmode == "Rmap" :
                cmatrix = spear(jarr, jarr, iarr  , iarr  , self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=False)
            elif self.spearmode == "Pmap" :
                cmatrix = spear(jarr, jarr, iarr  , iarr  , self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=True)
            elif self.spearmode == "SPmap" :
                cmatrix = spear(jarr, jarr, iarr+1, iarr+1, self.sigma, self.tau, self.lags, self.wids, self.scales, set_pmap=True) 
        # cholesky decomposed cmatrix to L, for which a C array is desired.
        L = np.empty(cmatrix.shape, order='C')
        L[:] = cholesky2(cmatrix) # XXX without the error report.
        # generate gaussian deviates y
        y = multivariate_normal(np.zeros(npt), np.identity(npt)) 
        # get x = L * y + u, where u is the mean of the light curve(s)
        x = np.dot(L, y) + marr
        # generate errors 
        # the way to get around peppering zeros is to generate deviates with unity std and multiply to earr.
        # XXX no covariance implemented here.
        e = earr * multivariate_normal(np.zeros(npt), np.identity(npt))  
        # add e 
        m = x + e 
        # unpack the data
        _jlist, _mlist, _elist = self._unpackdataarr(npt, nptlist, jarr, m, earr, iarr)
        zylclist_new = []
        for ilc in xrange(self.nlc_obs) :
            zylclist_new.append([_jlist[ilc], _mlist[ilc], _elist[ilc]])
        return(zylclist_new)

    def _combineddataarr(self, npt, nptlist, jlist, mlist, elist, ilist):
        """ Combine lists into ndarrays, taken directly from zylc.LightCurve.

        """
        jarr = np.empty(npt)
        marr = np.empty(npt)
        earr = np.empty(npt)
        iarr = np.empty(npt, dtype="int")
        start = 0
        for i, nptlc in enumerate(nptlist):
            jarr[start:start+nptlc] = jlist[i]
            marr[start:start+nptlc] = mlist[i]
            earr[start:start+nptlc] = elist[i]
            iarr[start:start+nptlc] = ilist[i]
            start = start+nptlc
        p = jarr.argsort()
        return(jarr[p], marr[p], earr[p], iarr[p])

    def _unpackdataarr(self, npt, nptlist, jarr, marr, earr, iarr):
        """ to reverse _combineddataarr.
        """
        jlist = []
        mlist = []
        elist = []
        for i in xrange(self.nlc_obs) :
            indxlc = (iarr == (i+1))
            # print marr.shape
            if np.sum(indxlc) != nptlist[i] :
                raise RuntimeError("iarr and data number do not match.")
            jlist.append(jarr[indxlc])
            mlist.append(marr[indxlc])
            elist.append(earr[indxlc])
        return(jlist, mlist, elist)

class PredictPmap(object):
    """ Predict light curves for Pmap, with data constraints.
    """
    def __init__(self, zydata, set_threading=False,  **covparams):
        """ PredictPmap object.

        Parameters
        ----------
        zydata: LightCurve object
            Observed light curve in LightCurve format.

        covparams: kwargs
            Parameters for the spear covariance function.

        """
        self.zydata = zydata
        self.covparams = covparams
        self.jd = self.zydata.jarr
        # has to be the true mean instead of the sample mean
        self.md = self.zydata.marr
        self.id = self.zydata.iarr
        self.blist = self.zydata.blist
        # variance
        self.vd = np.power(self.zydata.earr, 2.)
        # preparation
        self.set_threading = set_threading
        self._get_covmat(set_threading=set_threading)
        self._get_cholesky()
        self._get_cplusninvdoty()

    def mve_var(self, jwant, iwant):
        """ Generate the minimum variance estimate and its associated variance.

        Parameters
        ----------
        jwant: array_like
            Desired epochs for simulated light curve.

        iwant: array_like
            Desired ids for simulated light curve.

        Returns
        -------
        m: array_like
            Minimum variance estimate of the underlying signal.

        v: array_like
            Variance at simulated point.
        """
        m, v = self._fastpredict(jwant, iwant, set_threading=self.set_threading)
        for i in xrange(len(jwant)) :
            m[i] += self.blist[int(iwant[i])-1]
        return(m, v)

    def generate(self, zylclist) :
        """ Presumably zylclist has our input j, e, and i, and the values in m
        should be the mean.

        Parameters
        ----------
        zylclist: list of 3-list light curves 
            Pre-simulated light curves in zylclist, with the values in m-column as
            the light curve mean, and those in e-column as the designated errorbar.

        Returns
        -------
        zylclist_new: list of 3-list light curves
            Simulated light curves in zylclist.

        """
        nlc = len(zylclist)
        jlist = []
        mlist = []
        elist = []
        ilist = []
        for ilc, lclist in enumerate(zylclist):
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                if (np.min(msubarr) != np.max(msubarr)) : 
                    print("WARNING: input zylclist has inequal m elements in "+
                          "light curve %d, please make sure the m elements "+
                          "are filled with the desired mean of the mock "+
                          "light curves, now reset to zero"%ilc)
                    msubarr = msubarr * 0.0
                nptlc = len(jsubarr)
                # sort the date, safety
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
                ilist.append(np.zeros(nptlc, dtype="int")+ilc+1)
        zylclist_new = []
        for ilc in xrange(nlc) :
            m, v = self.mve_var(jlist[ilc], ilist[ilc])
            # no covariance considered here
            vcovmat = np.diag(v)
            if (np.min(elist[ilc]) < 0.0):
                raise RuntimeError("error for light curve %d should be either 0 or postive"%ilc)
            elif np.alltrue(elist[ilc]==0.0):
                set_error_on_mocklc = False
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) 
            else:
                set_error_on_mocklc = True
                ecovmat = np.diag(elist[ilc]*elist[ilc])
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) + multivariate_normal(np.zeros_like(m), ecovmat)
            zylclist_new.append([jlist[ilc], mlist[ilc], elist[ilc]])
        return(zylclist_new)

    def _get_covmat(self, set_threading=False) :
        if set_threading :
            self.cmatrix = spear_threading(self.jd,self.jd,self.id,self.id, set_pmap=True, **self.covparams)
        else :
            self.cmatrix = spear(self.jd,self.jd,self.id,self.id, set_pmap=True, **self.covparams)
        print("covariance matrix calculated")

    def _get_cholesky(self) :
        self.U = cholesky(self.cmatrix, nugget=self.vd, inplace=True, raiseinfo=True)
        print("covariance matrix decomposed and updated by U")

    def _get_cplusninvdoty(self) :
        # now we want cpnmatrix^(-1)*mag = x, which is the same as
        #    mag = cpnmatrix*x, so we solve this equation for x
        self.cplusninvdoty = chosolve_from_tri(self.U, self.md, nugget=None, inplace=False)

    def _fastpredict(self, jw, iw, set_threading=False) :
        """ jw : jwant
            iw : iwant
        """
        mw = np.zeros_like(jw)
        vw = np.zeros_like(jw)
        for i, (jwant, iwant) in enumerate(zip(jw, iw)):
            if set_threading :
                covar = spear_threading(jwant,self.jd,iwant,self.id, set_pmap=True, **self.covparams)
            else :
                covar = spear(jwant,self.jd,iwant,self.id, set_pmap=True, **self.covparams)
            cplusninvdotcovar = chosolve_from_tri(self.U, covar.T, nugget=None, inplace=False)
            if set_threading :
                vw[i] = spear_threading(jwant, jwant, iwant, iwant, set_pmap=True, **self.covparams)
            else :
                vw[i] = spear(jwant, jwant, iwant, iwant, set_pmap=True, **self.covparams)
            mw[i] = np.dot(covar, self.cplusninvdoty)
            vw[i] = vw[i] - np.dot(covar, cplusninvdotcovar)
        return(mw, vw)

class PredictSPmap(object):
    """ Predict light curves for SPmap, with data constraints.
    """
    def __init__(self, zydata, set_threading=False, **covparams):
        """ PredictPmap object.

        Parameters
        ----------
        zydata: LightCurve object
            Observed light curve in LightCurve format.

        covparams: kwargs
            Parameters for the spear covariance function.

        """
        self.zydata = zydata
        lags  = np.zeros(3)
        wids  = np.zeros(3)
        scales = np.ones(3)
        # XXX try following the format of arguments in pmap_model 
        lags[1]   = covparams["lag"]
        wids[1]   = covparams["wid"]
        scales[1] = covparams["scale"]
        sigma     = covparams["sigma"]
        tau       = covparams["tau"]
        self.covparams = {"lags" :  lags, "wids" : wids, "scales" : scales, "sigma" : sigma, "tau" : tau }
        self.jd = self.zydata.jarr
        # has to be the true mean instead of the sample mean
        self.md = self.zydata.marr
        self.id = self.zydata.iarr
        self.blist = self.zydata.blist
        # variance
        self.vd = np.power(self.zydata.earr, 2.)
        # preparation
        self.set_threading = set_threading
        self._get_covmat(set_threading=set_threading)
        self._get_cholesky()
        self._get_cplusninvdoty()

    def mve_var(self, jwant, iwant):
        """ Generate the minimum variance estimate and its associated variance.

        Parameters
        ----------
        jwant: array_like
            Desired epochs for simulated light curve.

        iwant: array_like
            Desired ids for simulated light curve.

        Returns
        -------
        m: array_like
            Minimum variance estimate of the underlying signal.

        v: array_like
            Variance at simulated point.
        """
        m, v = self._fastpredict(jwant, iwant, set_threading=self.set_threading)
        for i in xrange(len(jwant)) :
            m[i] += self.blist[int(iwant[i])-1]
        return(m, v)

    def generate(self, zylclist) :
        """ Presumably zylclist has our input j, e, and i, and the values in m
        should be the mean.

        Parameters
        ----------
        zylclist: list of 3-list light curves 
            Pre-simulated light curves in zylclist, with the values in m-column as
            the light curve mean, and those in e-column as the designated errorbar.

        Returns
        -------
        zylclist_new: list of 3-list light curves
            Simulated light curves in zylclist.

        """
        nlc = len(zylclist)
        if nlc !=  1 :
            raise InputError("SPmap only deal with a single light curve.")
        jlist = []
        mlist = []
        elist = []
        ilist = []
        # XXX keep this for loop even if nlc=1
        for ilc, lclist in enumerate(zylclist):
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                if (np.min(msubarr) != np.max(msubarr)) : 
                    print("WARNING: input zylclist has inequal m elements in "+
                          "light curve %d, please make sure the m elements "+
                          "are filled with the desired mean of the mock "+
                          "light curves, now reset to zero"%ilc)
                    msubarr = msubarr * 0.0
                nptlc = len(jsubarr)
                # sort the date, safety
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
                ilist.append(np.zeros(nptlc, dtype="int")+ilc+1)
        zylclist_new = []
        for ilc in xrange(nlc) :
            m, v = self.mve_var(jlist[ilc], ilist[ilc])
            # no covariance considered here
            vcovmat = np.diag(v)
            if (np.min(elist[ilc]) < 0.0):
                raise RuntimeError("error for light curve %d should be either 0 or postive"%ilc)
            elif np.alltrue(elist[ilc]==0.0):
                set_error_on_mocklc = False
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) 
            else:
                set_error_on_mocklc = True
                ecovmat = np.diag(elist[ilc]*elist[ilc])
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) + multivariate_normal(np.zeros_like(m), ecovmat)
            zylclist_new.append([jlist[ilc], mlist[ilc], elist[ilc]])
        return(zylclist_new)

    def _get_covmat(self, set_threading=False) :
        if set_threading :
            self.cmatrix = spear_threading(self.jd,self.jd,self.id+1,self.id+1, set_pmap=True, **self.covparams)
        else :
            self.cmatrix = spear(self.jd,self.jd,self.id+1,self.id+1, set_pmap=True, **self.covparams)
        print("covariance matrix calculated")

    def _get_cholesky(self) :
        self.U = cholesky(self.cmatrix, nugget=self.vd, inplace=True, raiseinfo=True)
        print("covariance matrix decomposed and updated by U")

    def _get_cplusninvdoty(self) :
        # now we want cpnmatrix^(-1)*mag = x, which is the same as
        #    mag = cpnmatrix*x, so we solve this equation for x
        self.cplusninvdoty = chosolve_from_tri(self.U, self.md, nugget=None, inplace=False)

    def _fastpredict(self, jw, iw, set_threading=False) :
        """ jw : jwant
            iw : iwant
        """
        mw = np.zeros_like(jw)
        vw = np.zeros_like(jw)
        for i, (jwant, iwant) in enumerate(zip(jw, iw)):
            if set_threading :
                covar = spear_threading(jwant,self.jd,iwant+1,self.id+1, set_pmap=True, **self.covparams)
            else :
                covar = spear(jwant,self.jd,iwant+1,self.id+1, set_pmap=True, **self.covparams)
            cplusninvdotcovar = chosolve_from_tri(self.U, covar.T, nugget=None, inplace=False)
            if set_threading :
                vw[i] = spear_threading(jwant, jwant, iwant+1, iwant+1, set_pmap=True, **self.covparams)
            else :
                vw[i] = spear(jwant, jwant, iwant+1, iwant+1, set_pmap=True, **self.covparams)
            mw[i] = np.dot(covar, self.cplusninvdoty)
            vw[i] = vw[i] - np.dot(covar, cplusninvdotcovar)
        return(mw, vw)

def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.

    from: http://www.scipy.org/Cookbook/SignalSmooth
    with some minor modifications.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """ 
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    # increment the original array 
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    # chop off the wings to maintain array size before return
    return(y[window_len-1:-window_len+1])

def mockme(zydata, covfunc="drw", rank="Full", mockname=None, shrinkerr=1.0, **covparams) :
    """ simulate a mock continuum light curve with the same sampling and error
    properties as the input data.
    """
    lcmean = zydata.blist[0]
    PS = PredictSignal(lcmean=lcmean, covfunc=covfunc, rank=rank, **covparams)
    jwant = zydata.jlist[0]
    ewant = zydata.elist[0]
    if shrinkerr != 1.0 :
        ewant = ewant*shrinkerr
    mwant = PS.generate(jwant, ewant=ewant, num=1)
    zymock_list = [[jwant, mwant, ewant],]
    if mockname is None :
        mockname = [zydata.names[0]+"_"+covfunc+"_mock"]
    zymock = LightCurve(zymock_list, names=[mockname,])
    return(zymock)

def test_PredictSpear():
    """
    """
    sigma = 0.2
    tau = 40.0
    llags = [10.0, 30.0]
    lwids = [ 2.0,  5.0]
    lscales = [ 0.5,  2.0]
    ps = PredictSpear(sigma, tau, llags, lwids, lscales, spearmode="Rmap")
    npt = 100
    jarr = np.linspace(0.0, 200.0, npt)
    marr = np.ones(npt)*10.0
    earr = np.zeros(npt)
    lcdat = [ [jarr, marr, earr], [jarr, marr, earr], [jarr, marr, earr],]
    lcnew = ps.generate(lcdat, set_threading=False)
    zylc = LightCurve(lcnew)
    zylc.plot()

if __name__ == "__main__":
    test_PredictSpear()
