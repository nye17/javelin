from gp import Mean, Covariance, observe, Realization, GPutils
from gp import NearlyFullRankCovariance, FullRankCovariance
from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri
import numpy as np
from numpy.random import normal, multivariate_normal
from cov import get_covfunc_dict
from spear import spear
from zylc import zyLC

np.set_printoptions(precision=3)


""" Generate random realizations based on the covariance function.
"""

def generateError(e, errcov=0.0):
    """ generate Gaussian errors.

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
    return(jl, ml)

class PredictRmap(object):
    """ Predict light curves for spear.
    """
    def __init__(self, zydata=None, **covparams):
        """ PredictRmap object.

        Parameters
        ----------
        zydata: zyLC object, optional
            Observed light curve in zyLC format, set to 'None' if no observation
            is done. Note that the true means of light curves should be
            subtracted (default: Done).

        covparams: kwargs
            Parameters for the spear covariance function.

        """
        self.zydata = zydata
        self.covparams = covparams
        self.jd = self.zydata.jarr
        # has to be the true mean instead of the sample mean
        self.md = self.zydata.marr
        self.id = self.zydata.iarr
        # variance
        self.vd = np.power(self.zydata.earr, 2.)
        # preparation
        self._get_covmat()
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
        return(self._fastpredict(jwant, iwant))

    def generate(self, zylclist) :
        """ Presumably zylclist has our input j, e, and i, and the values in m
        should be the mean.

        Parameters
        ----------
        zylclist: list of 3-list light curves 
            Pre-simulated light curves in zylclist, with the values in m-column as
            the light curve mean.

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
                raise RuntimeError("error for light curve %d should be either"+
                        " 0 or postive"%ilc)
            elif np.alltrue(elist[ilc]==0.0):
                set_error_on_mocklc = False
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) 
            else:
                set_error_on_mocklc = True
                ecovmat = np.diag(elist[ilc]*elist[ilc])
                mlist[ilc] = mlist[ilc] + multivariate_normal(m, vcovmat) + multivariate_normal(np.zeros_like(m), ecovmat)
            zylclist_new.append([jlist[ilc], mlist[ilc], elist[ilc]])
        return(zylclist_new)

    def _get_covmat(self) :
        self.cmatrix = spear(self.jd,self.jd,self.id,self.id, **self.covparams)
        print("covariance matrix calculated")

    def _get_cholesky(self) :
        self.U = cholesky(self.cmatrix, nugget=self.vd, inplace=True, raiseinfo=True)
        print("covariance matrix decomposed and updated by U")

    def _get_cplusninvdoty(self) :
        # now we want cpnmatrix^(-1)*mag = x, which is the same as
        #    mag = cpnmatrix*x, so we solve this equation for x
        self.cplusninvdoty = chosolve_from_tri(self.U, self.md, nugget=None, inplace=False)

    def _fastpredict(self, jw, iw) :
        """ jw : jwant
            iw : iwant
        """
        mw = np.zeros_like(jw)
        vw = np.zeros_like(jw)
        for i, (jwant, iwant) in enumerate(zip(jw, iw)):
            covar = spear(jwant,self.jd,iwant,self.id, **self.covparams)
            cplusninvdotcovar = chosolve_from_tri(self.U, covar.T, nugget=None, inplace=False)
            vw[i] = spear(jwant, jwant, iwant, iwant, **self.covparams)
            mw[i] = np.dot(covar, self.cplusninvdoty)
            vw[i] = vw[i] - np.dot(covar, cplusninvdotcovar)
        return(mw, vw)


class PredictSignal(object):
    """
    Predict continuum light curves.
    """
    def __init__(self, zydata=None, lcmean=0.0, covfunc="drw", 
            rank="Full", **covparams):
        """ PredictSignal object for simulating continuum light curves.

        Parameters
        ----------
        zydata: zyLC object, optional
            Observed light curve in zyLC format, set to 'None' if no observation
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
        elif rank is "NearlyFull" :
            self.C  = NearlyFullRankCovariance(**covfunc_dict)
        # observe zydata
        if zydata is None :
            print("No *zydata* Observed, Unconstrained Realization")
        else:
            print("Observed *zydata*, Constrained Realization")
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


def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.

    from: http://www.scipy.org/Cookbook/SignalSmooth
    
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
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return(y)


