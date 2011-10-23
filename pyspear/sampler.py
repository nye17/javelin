
from pylab import *
from pyspear.gp import Mean, Covariance, GPSubmodel, GPEvaluationGibbs, gpplots, FullRankCovariance
#from pyspear.gp import Mean, Covariance, observe, Realization, GPutils
from pyspear.gp.cov_funs import matern, quadratic, gaussian, pow_exp, sphere

import numpy as np
import pymc as pm 

class PowExpSampler(pm.MCMC):
    def __init__(self, name, jdata, mdata, edata):
        self.name  = name
        self.jdata = jdata
        self.mdata = mdata
        self.edata = edata

        npt = len(jdata)

        lcmean_obs, lcstd_obs = self.get_lcstat()
        errmean_obs = np.mean(edata)
        rx = jdata.max() - jdata.min()
        ry = mdata.max() - mdata.min()
        self.plot_x = np.linspace(jdata.min()-0.2*rx, jdata.max()+0.2*rx,100)

        # ============
        # = The mean =
        # ============

        lcmean = pm.Uniform('lcmean', lcmean_obs-2*lcstd_obs, lcmean_obs+2*lcstd_obs,value=lcmean_obs)
        def constant(x, val):
            return(np.zeros(x.shape[:-1],dtype=float) + val)
        @pm.deterministic
        def M(lcmean=lcmean):
            return(Mean(constant, val=lcmean))

        # ==================
        # = The covariance =
        # ==================

        sigma = pm.Exponential('sigma',7.e-5, value=lcstd_obs)
        tau   = pm.Exponential('tau',  4.e-3, value=rx/2.0)
        nu    = pm.Uniform('nu', 0, 2, value=1.0)
        @pm.deterministic
        def C(nu=nu, sigma=sigma, tau=tau):
#            C = Covariance(pow_exp.euclidean, pow=nu, amp=sigma, scale=tau)
            C = FullRankCovariance(pow_exp.euclidean, pow=nu, amp=sigma, scale=tau)
            return(C)

        # ===================
        # = The GP submodel =
        # ===================

        PES = GPSubmodel(self.name + '.PES', M, C, mesh=jdata)

        # ============
        # = The data =
        # ============

        errcov = pm.Uniform("errcov", lower=0.0, upper=0.5, value=0.0)
        @pm.deterministic
        def phvar(errcov=errcov):
            ediag = np.diag(edata*edata)
            temp1 = np.repeat(edata, npt).reshape(npt,npt)
            temp2 = (temp1*temp1.T - ediag)*errcov
            phvar = ediag + temp2
            return(phvar)

        @pm.observed
        @pm.stochastic
        def variability(value=mdata, mu=PES.f_eval, var=phvar):
            _p = pm.mv_normal_cov_like(value, mu, var)
            return(_p)

        pm.MCMC.__init__(self, locals())
        
        self.use_step_method(GPEvaluationGibbs, PES, phvar, variability)

        self.lcmean = lcmean 
        self.sigma = sigma
        self.tau = tau
        self.nu = nu
        self.errcov = errcov

    def get_lcstat(self):
        _lcmean =  np.average(self.mdata, weights=np.power(self.edata, -2))
        _lcstd  =  np.std(self.mdata)
        return(_lcmean, _lcstd)

        self.lcrms_obs  = self.get_lcrms()

    def plot_traces(self):
        for object in [self.lcmean, self.sigma, self.tau, self.nu, self.errcov]:
            try:
                y=object.trace()
            except:
                print(object.__name__)
                break
            print(len(y))
            try:
                figure()
                plot(y)
                title(object.__name__)
                show()
            except:
                print("plot trace failed")


    def plot_PES(self):
        f_trace = self.PES.f.trace()
        figure()
        hold('on')
        gpplots.plot_GP_envelopes(self.PES.f, self.plot_x)

        for i in range(3):
            plot(self.plot_x, f_trace[i](self.plot_x), label='draw %i'%i)

        plot(self.jdata, self.mdata, 'k.', label='data', markersize=8)
        legend(loc=0)
        axis([self.plot_x[0], self.plot_x[-1], self.mdata.min()*0.5, self.mdata.max()*1.5])
        show()

    def plot_post(self):
        pm.Matplot.plot(self)
