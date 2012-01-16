from gp import Mean, Covariance, observe, Realization, GPutils
import numpy as np
from numpy.random import normal, multivariate_normal
from covariance import get_covfunc_dict


""" Generate random realizations based on the covariance function.
"""


class Predict(object):
    """
    Predict light curves at given input epoches in two possible scenarios.
    1) random realizations of the underlying process defined by both 
    mean and covariance.
    2) constrained realizations of the underlying process defined by 
    both mean and covariance, and observed data points.
    """
    def __init__(self, lcmean=0.0, jdata=None, mdata=None, edata=None,
            covfunc="pow_exp", **covparams):
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
        self.C  = Covariance(covfunc_dict)

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

        ediag = np.diag(ewant*ewant)
        temp1 = np.repeat(ewant, nwant).reshape(nwant,nwant)
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




def test_Predict():
    from pylab import fill, plot, show
    jdata = np.array([25., 100, 175.])
    mdata = np.array([0.7, 0.1, 0.4])
    edata = np.array([0.07, 0.02, 0.05])
    j = np.arange(0, 200, 1)
    P = Predict(jdata=jdata, mdata=mdata, edata=edata, nu=1.0)
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

def test_simlc():
    covfunc = "kepler_exp"
    tau, sigma, nu = (10.0, 2.0, 0.1)
    j = np.linspace(0., 200, 4096)
    lcmean = 10.0
#    emean  = lcmean*0.05
    emean  = lcmean*0.00
    print("observed light curve mean mag is %10.3f"%lcmean)
    print("observed light curve mean err is %10.3f"%emean)
    P = Predict(lcmean=lcmean, covfunc=covfunc, tau=tau, sigma=sigma, nu=nu)
    ewant = emean*np.ones_like(j)
    mwant = P.generate(j, nlc=1, ewant=ewant, errcov=0.0)
    np.savetxt("mock2.dat", np.vstack((j, mwant, ewant)).T)

if __name__ == "__main__":    
    test_simlc()
