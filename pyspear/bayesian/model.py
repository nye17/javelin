import numpy as np
import pymc as pm
from prh import PRH

def make_model_powexp(zydata, set_cskprior=False):
    cadence = zydata.cont_cad
    rx = zydata.rj
    ry = zydata.marr.max() - zydata.marr.min()
    #-------
    # priors
    #-------
    # sigma
    if set_cskprior:
        @pm.stochastic
        def sigma(value=ry/4.):
            def logp(value):
                if (value > 0.0):
                   return(-np.log(value))
                elif(value < 0.0):
                    return(-np.Inf)
    else:
        variance = pm.InverseGamma('variance' , alpha=2., beta=(ry/4.0)**2., value=(ry/4.0)**2.)
        @pm.deterministic
        def sigma(name="sigma", variance=variance):
            return(np.sqrt(variance))

    # tau
    if set_cskprior:
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
#        tau   = pm.InverseGamma('tau' , alpha=2., beta=rx/6.0, value=rx/6.0)
        tau   = pm.InverseGamma('tau' , alpha=2., beta=np.sqrt(rx*cadence), value=rx/6.0)
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


