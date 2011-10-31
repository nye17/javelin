import numpy as np
import pymc as pm
import pickle
from prh import PRH

from data import get_data
from model import make_model_powexp

from pyspear.traceless import TraceLess 

def pretty_array(x):
    return('[%s]' % ', '.join('%.2f' % x_i for x_i in x))

def runMCMC(model, txtdb, iter=10000, burn=1000, thin=2, verbose=0,
        set_geweke=False, set_sumplot=False):
    M = pm.MCMC(model, db='txt', dbname=txtdb, dbmode="w")
#    M.use_step_method(pm.AdaptiveMetropolis,[M.tau, M.invsigsq, M.sigma, M.nu])
    M.use_step_method(pm.AdaptiveMetropolis,[M.tau, M.sigma, M.nu])
    print("**************************************")
    print("Initial sigma, tau, nu")
    print(pretty_array([M.sigma.value, M.tau.value, M.nu.value]))
    M.sample(iter=iter, burn=burn, thin=thin, verbose=verbose)
    print('sigma',np.median(M.trace('sigma')[:]))
    print('tau',np.median(M.trace('tau')[:]))
    print('nu',np.median(M.trace('nu')[:]))
    M.db.commit()
    if set_sumplot:
        pm.Matplot.plot(M)
    if set_geweke:
        scores = pm.geweke(M, intervals=20)
        pm.Matplot.geweke_plot(scores)
    print("**************************************")

def anaMCMC(resource, db='txt'):
    T = TraceLess(resource, db=db)
    cred1s = 0.683
    print("**************************************")
    print(T)
    print("--------------------------------------")
    retdict = {}
    for para in ['sigma', 'tau', 'nu']:
        print(para+":")
        medpar = T.median(para)
        print("median: %.2f" % medpar)
        hpdlow, hpdupp  = T.hpd(para, cred=cred1s)
        print("1-sigma HPD: %.2f %.2f" % (hpdlow, hpdupp))
        print("mean: %.2f std: %.2f" % (T.mean(para), T.std(para)))
        print("--------------------------------------")
        retdict[para] = [medpar, hpdlow, hpdupp]
    print("MCMC Database Analysis Ends")
    print("**************************************")
    return(retdict)

def getPlike(zydata, par):
    print("**************************************")
    print("Input  sigma, tau, nu")
    print(pretty_array(par))
    prh = PRH(zydata, covfunc="pow_exp",
                           sigma=par[0], tau=par[1], nu=par[2])
    out = prh.loglike_prh()
    print("--------------------------------------")
    print("Output logL, -chi2/2, complexity, drift, [q]")
    print(pretty_array(out))
    print("**************************************")
    return(out)

def runMAP(model):
    M = pm.MAP(model)
    print("**************************************")
    print("Initial sigma, tau, nu")
    print(pretty_array([M.sigma.value, M.tau.value, M.nu.value]))
    M.fit()
    print("--------------------------------------")
    print("Bestfit sigma, tau, nu")
    print(pretty_array([M.sigma.value, M.tau.value, M.nu.value]))
    print("**************************************")
    parbestfit = [
     M.sigma.value,
     M.tau.value,
     M.nu.value,
    ]
    return(parbestfit)


if __name__ == "__main__":    
    lcfile  = "dat/mock_l100c1_t10s2n0.5.dat"
    zydata  = get_data(lcfile)
#    testout = getPlike(zydata, [2., 10., 0.5])

    model  = make_model_powexp(zydata, set_csktauprior=False)
##    model  = make_model_powexp(zydata, set_csktauprior=True)
    runMAP(model)


    runMCMC(model, "/home/mitchell/yingzu/petest", iter=50000, burn=5000, thin=2, verbose=0)
#    retdict = anaMCMC("~/petest", db='txt')
#    print(retdict)
