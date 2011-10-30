import numpy as np
import pymc as pm
from prh import PRH

def make_model_powexp(zydata, set_csktauprior=False):
    cadence = zydata.cont_cad
    rx = zydata.rj
    ry = zydata.marr.max() - zydata.marr.min()
    #-------
    # priors
    #-------
    # sigma
    invsigsq = pm.Gamma('invsigsq', alpha = 2., beta = 1./(ry/4.)**2)
    @pm.deterministic
    def sigma(name="sigma", invsigsq=invsigsq):
        return(1./np.sqrt(invsigsq))
    # tau
    if set_csktauprior:
        # double-lobed log prior on tau, stemmed from CSK's original code
        @pm.stochastic
        def tau(value=30.0):
            def logp(value):
                if (10000 > value > 1.0*cadence):
                    return(-np.log(value/(1.0*cadence)))
                elif(0.0 < value <= 1.0*cadence):
                    return(-np.log(1.0*cadence/value))
                else:
                    return(-np.Inf)
    else:
        # inverse gamma prior on tau, penalty on extremely small or large scales.
        tau   = pm.InverseGamma('tau' , alpha=2., beta=rx/6.0, value=rx/6.0)
    # nu
    # uniform prior on nu
    nu    = pm.Uniform('nu', 0, 2, value=1.0)

    #-------
    # model
    #-------
    guess = [2., 30., 0.5]
    @pm.stochastic(observed=True)
    def model_powexp(value=guess, 
                     sigma=sigma, tau=tau, nu=nu):
        par=[sigma, tau, nu]
        prh = PRH(zydata, covfunc="pow_exp", 
                               sigma=par[0], tau=par[1], nu=par[2])
        out = prh.loglike_prh()
        return(out[0])

    return(locals())


