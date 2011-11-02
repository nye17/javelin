import numpy as np
import pymc as pm
import pickle
from prh import PRH
import os.path

from data import get_data
from model import make_model_powexp

from pyspear.traceless import TraceLess 

def pretty_array(x):
    return('[%s]' % ', '.join('%.2f' % x_i for x_i in x))

def runMCMC(model, txtdb, iter=10000, burn=1000, thin=2, verbose=0,
        set_geweke=False, set_sumplot=False):
    M = pm.MCMC(model, db='txt', dbname=txtdb, dbmode="w")
    tosample = []
    if M.fixsigma is not None:
        tosample.append(M.sigma)
    if M.fixtau   is not None:
        tosample.append(M.tau)
    if M.fixnu    is not None:
        tosample.append(M.nu)
    if M.invsigsq is not None:
        tosample.append(M.invsigsq)
    M.use_step_method(pm.AdaptiveMetropolis, tosample)
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
    resource = os.path.expanduser(resource)
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
    tovary = getValues(M)
    print("**************************************")
    print("Initial sigma, tau, nu")
    print(pretty_array(tovary))
    M.fit()
    tovary = getValues(M)
    print("--------------------------------------")
    print("Bestfit sigma, tau, nu")
    print(pretty_array(tovary))
    print("**************************************")
    return(tovary)

def getValues(M):
    tovary = []
    if is_number(M.use_sigprior):
        tovary.append(M.sigma)
    else:
        tovary.append(M.sigma.value)
    if is_number(M.use_tauprior):
        tovary.append(M.tau)
    else:
        tovary.append(M.tau.value)
    if is_number(M.use_nuprior):
        tovary.append(M.nu)
    else:
        tovary.append(M.nu.value)
    return(tovary)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False




if __name__ == "__main__":    
#    lcfile  = "dat/mock_l100c1_t10s2n0.5.dat"
#    lcfile  = "dat/t100n0_6.dat"
    lcfile  = "dat/t1000n0_6.dat"
    print("reading %s"%lcfile)
    zydata  = get_data(lcfile)
#    testout = getPlike(zydata, [2., 10., 0.5])

#    model   = make_model_powexp(zydata, use_sigprior="CSK", use_tauprior="CSK", use_nuprior="Uniform")
    model   = make_model_powexp(zydata, use_sigprior="None", use_tauprior="None", use_nuprior="Uniform")
    bestpar = runMAP(model)
    testout = getPlike(zydata, bestpar)

#    model   = make_model_powexp(zydata, use_sigprior="CSK", use_tauprior=10.0, use_nuprior="Uniform")
#    bestpar = runMAP(model)
#    testout = getPlike(zydata, bestpar)

#    model   = make_model_powexp(zydata, use_sigprior="CSK", use_tauprior=1000.0, use_nuprior="Uniform")
#    bestpar = runMAP(model)
#    testout = getPlike(zydata, bestpar)

#    runMCMC(model, "/home/mitchell/yingzu/tmp/petest", iter=2000, burn=0, thin=1, verbose=0)
#    retdict = anaMCMC("~/tmp/petest", db='txt')
#    print(retdict)
