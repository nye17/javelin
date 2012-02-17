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


def predictLine(jc, mc, lag, wid, scale, mc_mean=0.0, ml_mean=0.0):
    """ Convolve with the top-hat kernel to get line light curve, however, the
    input continuum signal has to be dense enough and regularly sampled.
    """
    djc = jc[1:] - jc[:-1]
    if (np.abs(np.min(djc) - np.max(djc))>0.01) :
        raise RuntimeError("input jc has to be regularly sampled %.10g %.10g"%(np.min(djc) , np.max(djc)))
    # scale time unit to djc
    junit = djc[0]
    window_len = np.floor(0.5*wid/junit)
    # continuum signal 
    sc = mc - mc_mean
    if wid < 0.0 :
        wid = np.abs(wid)
        print("WARNING: negative wid? reset to abs(wid) %10.4f"%wid)
    if scale < 0.0 :
        scale = np.abs(scale)
        print("WARNING: negative scale? reset to abs(scale) %10.4f"%scale)
    sl = smooth(sc,window_len=window_len,window='flat')
    ml = ml_mean + sl*scale
    jl = jc + lag
    return(jl, ml)

def generateErrorTerm(e):
    ecovmat = np.diag(e*e)
    et = multivariate_normal(np.zeros_like(e), ecovmat)
    return(et)

class PredictRmap(object):
    """ Predict light curves for spear.
    """
    def __init__(self, zydata=None, **covparams):
        self.zydata = zydata
        self.covparams = covparams
        self.jd = self.zydata.jarr
        # has to be the true mean instead of the samle mean
        self.md = self.zydata.marr
        self.id = self.zydata.iarr
        self.vd = np.power(self.zydata.earr, 2.)
        # preparation
        self._get_covmat()
        self._get_cholesky()
        self._get_cplusninvdoty()

    def generate(self, zylclist) :
        """ presumably zylclist has our input j, e, and i, and the values in m
         should be the mean."""
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

    def mve_var(self, jwant, iwant):
        return(self._fastpredict(jwant, iwant))

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




class Predict(object):
    """
    Predict light curves at given input epoches in two possible scenarios.
    1) random realizations of the underlying process defined by both 
    mean and covariance.
    2) constrained realizations of the underlying process defined by 
    both mean and covariance, and observed data points.
    """
    def __init__(self, lcmean=0.0, jdata=None, mdata=None, edata=None,
            covfunc="pow_exp", rank="Full", **covparams):
        try :
            const = float(lcmean)
            meanfunc = lambda x: const*(x*0.0+1.0)
            self.M = Mean(meanfunc)
        except ValueError:
            if isinstance(lcmean, Mean):
                self.M = lcmean
            else:
                raise RuntimeError("lcmean is neither a Mean obj or a const")
        
        covfunc_dict = get_covfunc_dict(covfunc, **covparams)
        if rank is "Full" :
            self.C  = FullRankCovariance(**covfunc_dict)
        elif rank is "NearlyFull" :
            self.C  = NearlyFullRankCovariance(**covfunc_dict)

        if ((jdata is not None) and (mdata is not None) and (edata is not None)):
            print("Constrained Realization...")
            observe(self.M, self.C, obs_mesh=jdata, obs_V = edata, obs_vals = mdata)
        else:
            print("No Data Input or Some of jdata/mdata/edata Are None")
            print("Unconstrained Realization...")

    def generate(self, jwant, ewant=0.0, nlc=1, errcov=0.0):
        if (np.min(ewant) < 0.0):
            raise RuntimeError("ewant should be either 0  or postive")
        elif np.alltrue(ewant==0.0):
            set_error_on_mocklc = False
        else:
            set_error_on_mocklc = True

        nwant = len(jwant)

        if np.isscalar(ewant):
            e = np.zeros(nwant) + ewant
        elif len(ewant) == nwant:
            e = ewant
        else:
            raise RuntimeError("ewant should be either a const or array with same shape as jwant")

        ediag = np.diag(e*e)
        temp1 = np.repeat(e, nwant).reshape(nwant,nwant)
        temp2 = (temp1*temp1.T - ediag)*errcov
        ecovmat = ediag + temp2

        if nlc == 1:
            f = Realization(self.M, self.C)
            mwant = f(jwant)
            if set_error_on_mocklc:
                mwant = mwant + multivariate_normal(np.zeros(nwant), ecovmat)
            return(mwant)
        else:
            mwant_list = []
            for i in xrange(nlc):
                f = Realization(self.M, self.C)
                mwant = f(jwant)
                mwant = mwant + multivariate_normal(np.zeros(nwant), ecovmat)
                mwant_list.append(mwant)
            return(mwant_list)

    def mve_var(self, jwant):
        m, v = GPutils.point_eval(self.M, self.C, jwant)
        return(m,v)


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
    return y


def genSingle(covfunc, zydata=None, **covparams):
    pass

def genTophat():
    pass



def test_Predict():
    from pylab import fill, plot, show
    zydata = test_simsingle(set_plot=False)
    j = np.arange(0, 200, 1)
    P = Predict(jdata=zydata.jarr, mdata=zydata.marr, edata=zydata.earr, covfunc="drw", tau=10.0, sigma=0.2)
    mve, var = P.mve_var(j)
    sig = np.sqrt(var)
    x=np.concatenate((j, j[::-1]))
    y=np.concatenate((mve-sig, (mve+sig)[::-1]))
    fill(x,y,facecolor='.8',edgecolor='1.')
    plot(j, mve, 'k-.')
    mlist = P.generate(j, nlc=3, ewant=0.0)
    for m in mlist:
        plot(j, m)
    show()

def test_simsingle(set_plot=True):
    covfunc = "drw"
    tau, sigma = (20.0, 2.0)
    jwant = np.linspace(0., 200, 50)
    lcmean  = 10.0
    errfrac = 0.05
    emean   = lcmean*errfrac
    P = Predict(lcmean=lcmean, covfunc=covfunc, tau=tau, sigma=sigma)
    ewant = emean*np.ones_like(jwant)
    mwant = P.generate(jwant, ewant=ewant)
    zylclist = [[jwant, mwant, ewant],]
    zydata = zyLC(zylclist, names=["continuum",], set_subtractmean=True, qlist=None)
    if set_plot :
        zydata.plot()
    return(zydata)

def test_PredictRmap():
    zydata = test_simtophat(set_plot=True, stride=5)
    tau, sigma = (20.0, 2.0)
    lag, wid, scale = (20.0, 10.0, 0.5)
    lags   = [0.0, lag]
    wids   = [0.0, wid]
    scales = [1.0, scale]
    P = PredictRmap(zydata=zydata, sigma=sigma, tau=tau, lags=lags, wids=wids, scales=scales)
    mcmean  = 10.0
    mlmean  =  5.0
    errfrac = 0.05
    jc = np.linspace(0., 100, 100)
    jl = jc
    mc = np.zeros_like(jc)+mcmean
    ml = np.zeros_like(jl)+mlmean
    ec = mc*errfrac
    el = ml*errfrac
    zylclist = [[jc, mc, ec], [jl, ml, el]]
    zylclist_new = P.generate(zylclist)
    zydata_new = zyLC(zylclist_new, names=["continuum", "line"], set_subtractmean=True, qlist=None)
    zydata_new.plot()

def test_simtophat(set_plot=True, stride=None):
    covfunc = "drw"
    tau, sigma = (20.0, 2.0)
    jc = np.linspace(0., 100, 500)
    mcmean  = 10.0
    mlmean  =  5.0
    errfrac = 0.05
    P = Predict(lcmean=mcmean, covfunc=covfunc, tau=tau, sigma=sigma)
    sc = P.generate(jc, ewant=0.0)
    lag, wid, scale = (20.0, 10.0, 0.5)
    jl, sl = predictLine(jc, sc, lag, wid, scale, mc_mean=mcmean, ml_mean=mlmean)
    ec = np.zeros_like(sc) + mcmean*errfrac
    el = np.zeros_like(sl) + mlmean*errfrac
    mc = sc + generateErrorTerm(ec)
    ml = sl + generateErrorTerm(el)
    if stride is None :
        zylclist = [[jc, mc, ec], [jl, ml, el]]
    else :
        indx = np.arange(0, 500, stride)
        zylclist = [[jc[indx], mc[indx], ec[indx]], [jl[indx], ml[indx], el[indx]]]
    
    zydata = zyLC(zylclist, names=["continuum", "line"], set_subtractmean=True, qlist=None)
    if set_plot :
        zydata.plot()
    return(zydata)


if __name__ == "__main__":    
    import matplotlib.pyplot as plt
#    test_simsingle()
#    test_Predict()
#    test_simtophat()
    test_PredictRmap()
    plt.show()
    pass
