#Last-modified: 30 Nov 2011 02:08:44 PM
import numpy as np
import pickle
import os.path
import pymc as pm

from prh import PRH
from data import get_data
from model import make_model_cov3par
from peakdetect import peakdet1d, peakdet2d

import matplotlib.pyplot as plt

from pyspear.traceless import TraceLess 

def runMCMC(model, txtdb, iter=10000, burn=1000, thin=2, verbose=0,
                          set_geweke=False, set_sumplot=False):
    """ running MCMC chains for the input model and save into a txt database.
    """
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
    """ load and analysis MCCM database.
    """
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

def getPlike(zydata, par, covfunc, set_verbose=False):
    """ Calculate the log marginal likelihood of data given input paramters.
    """
    if set_verbose:
        print("**************************************")
        print("Input  sigma, tau, nu")
        print(pretty_array(par))
    prh = PRH(zydata, covfunc=covfunc, sigma=par[0], tau=par[1], nu=par[2])
    out = prh.loglike_prh()
    if set_verbose:
        print("--------------------------------------")
        print("Output logL, -chi2/2, complexity, drift, [q]")
        print(pretty_array(out))
        print("**************************************")
    return(out)

def runMAP(model, set_verbose=True):
    """ Running MAP ananlysis.
    """
    M = pm.MAP(model)
    tovary = getValues(M)
    if set_verbose:
        print("**************************************")
        print("Initial sigma, tau, nu")
        print(pretty_array(tovary))
    if set_verbose:
        verbose=1
    else:
        verbose=0
    M.fit(verbose=verbose)
    tovary = getValues(M)
    if set_verbose:
        print("--------------------------------------")
        print("Bestfit sigma, tau, nu")
        print(pretty_array(tovary))
        print("**************************************")
    return(tovary)

def getValues(M):
    """ Get current paramter values of the model.
    """
    tovary = []
    if is_number(M.use_sigprior):
        tovary.append(M.sigma)
    else:
        tovary.append(np.atleast_1d(M.sigma.value)[0])
    if is_number(M.use_tauprior):
        tovary.append(M.tau)
    else:
        tovary.append(np.atleast_1d(M.tau.value)[0])
    if is_number(M.use_nuprior):
        tovary.append(M.nu)
    else:
        tovary.append(np.atleast_1d(M.nu.value)[0])
    return(tovary)

def varying_tau(output, zydata, tauarray, covfunc="pow_exp", fixednu=None, set_verbose=False):
    """ grid optimization along tau axis.
    """
#    result = []
    f=open(output, "w")
    if fixednu is None:
        use_nuprior="Uniform"
    elif is_number(fixednu):
        use_nuprior=fixednu
    else:
        raise RuntimeError("no such fixednu option")
    for tau in tauarray:
        print("tau: %10.5f"%tau)
        model   = make_model_cov3par(zydata, covfunc=covfunc,
                use_sigprior="None", use_tauprior=tau, use_nuprior=use_nuprior)
        bestpar = list(runMAP(model, set_verbose=set_verbose))
        testout = list(getPlike(zydata, bestpar,  covfunc, set_verbose=set_verbose))
#        result.append(" ".join(format(r, "10.4f") for r in bestpar+testout)+"\n")
        f.write(" ".join(format(r, "10.4g") for r in bestpar+testout)+"\n")
#    f.write("".join(result))
    f.close()

def varying_tau_nu(output, zydata, tauarray, nuarray, covfunc="pow_exp", set_verbose=False):
    """ grid optimization along both tau and nu axes.
    """
    dim_tau = len(tauarray)
    dim_nu  = len(nuarray)
    f=open(output, "w")
    # write dims into the header string
    header = " ".join(["#", str(dim_tau), str(dim_nu), "\n"])
    f.write(header)
    for tau in tauarray:
        print("tau: %10.5f"%tau)
        for nu in nuarray:
            print("_______________  nu: %10.5f"%nu)
            model   = make_model_cov3par(zydata, covfunc=covfunc, 
                    use_sigprior="None", use_tauprior=tau, use_nuprior=nu)
            bestpar = list(runMAP(model, set_verbose=set_verbose))
            testout = list(getPlike(zydata, bestpar, covfunc=covfunc, set_verbose=set_verbose))
            f.write(" ".join(format(r, "10.4g") for r in bestpar+testout)+"\n")
            f.flush()
    f.close()

def read_grid_tau_nu(input):
    """ read the grid file
    """ 
    print("reading from %s"%input)
    f = open(input, "r")
    dim_tau, dim_nu = [int(r) for r in f.readline().lstrip("#").split()]
    print("dim of tau: %d"%dim_tau)
    print("dim of  nu: %d"%dim_nu)
#    0.0915     1.0000     0.0000   956.2928  -238.4517  1197.6332    -2.8886     0.0011
    sigma, tau, nu, loglike, chi2, complexity, drift, q = np.genfromtxt(f, unpack=True)
    f.close()
    retdict = {
               'sigma'          :        sigma.reshape(dim_tau, dim_nu),
               'tau'            :          tau.reshape(dim_tau, dim_nu),
               'nu'             :           nu.reshape(dim_tau, dim_nu),
               'loglike'        :      loglike.reshape(dim_tau, dim_nu),
               'chi2'           :         chi2.reshape(dim_tau, dim_nu),
               'complexity'     :   complexity.reshape(dim_tau, dim_nu),
               'drift'          :        drift.reshape(dim_tau, dim_nu),
               'q'              :            q.reshape(dim_tau, dim_nu),
              }
    return(retdict)

def show_loglike_map(x, y, z, ax, cax=None, 
                     set_contour=True, clevels=None,
                     vmin=None, vmax=None, xlabel=None, ylabel=None, zlabel=None, 
                     set_normalize=True, peakpos=None, cmap='jet'):
    """ Display the 2D likelihood maps, assuming xmesh and ymesh are on regular grids.
    """
    xmin,xmax,ymin,ymax = np.min(x),np.max(x),np.min(y),np.max(y)
    extent = (xmin,xmax,ymin,ymax)
    if set_normalize:
        zmax = np.max(z)
        z    = z - zmax
    if vmin is None:
        vmin = z.min()
    if vmax is None:
        vmax = z.max()
    im = ax.imshow(z, origin='lower', vmin=vmin, vmax=vmax,
                      cmap=cmap, interpolation="nearest", aspect="auto", extent=extent)
    if set_contour:
        if clevels is None:
            sigma3,sigma2,sigma1 = 11.8/2.0,6.17/2.0,2.30/2.0
            levels = (vmax-sigma1, vmax-sigma2, vmax-sigma3)
        else:
            levels = clevels
        ax.set_autoscale_on(False)
        cs = ax.contour(z,levels, hold='on',colors='k',
                          origin='lower',extent=extent)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if peakpos:
        ax.set_autoscale_on(False)
        ax.plot(x[peakpos],y[peakpos], 'wx', markersize=20)
    if cax:
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_ylabel(zlabel)


def pretty_array(x):
    """ Return a string from list or array for nice print format.
    """
    return('[%s]' % ', '.join('%.2f' % x_i for x_i in x))

def is_number(s):
    """ Check if s is a number or not.
    """
    try:
        float(s)
        return(True)
    except ValueError:
        return(False)

def ogle_example_1():
    """ Use mock light curves from OGLE as an example of 1-D grid optimization, everything
    is in data/OGLE_example directory.
    """
    tauarray = np.power(10.0, np.arange(0, 4.2, 0.1))
    truetau  = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
    truenu   = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])

    for ttau in truetau:
        for tnu in truenu:
            lcfile = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".dat"
            print("reading %s"%lcfile)
            zydata  = get_data(lcfile)
            record = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_hires.dat"
            print("writing %s"%record)
            varying_tau(record, zydata, tauarray)
    print("done!")

def ogle_example_2():
    """ another OGLE example, testing 2D grid optimization using 'varying_tau_nu'.
    """
    tauarray = np.power(10.0, np.arange(0, 4.2, 0.1))
    print(tauarray)
    nuarray  = np.arange(0.0, 2.0, 0.2)
    print(nuarray)
    ttau = 200.0
    tnu = 1.4
    lcfile = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".dat"
    print("reading %s"%lcfile)
    zydata  = get_data(lcfile)
    record = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".test2dgrid.dat"
    print("writing %s"%record)
    varying_tau_nu(record, zydata, tauarray, nuarray, covfunc="pow_exp", set_verbose=False)
    print("done!")

def ogle_example_3():
    """ yet another example, testing 2D grid visualization and peak finding.
    """
    ttau = 200.0
    tnu = 1.4
    record   = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".test2dgrid.dat"
    griddict = read_grid_tau_nu(record)
    print(griddict.keys())
    fig = plt.figure(figsize=(6, 10))
    ax1  = fig.add_axes([0.2, 0.1, 0.55, 0.25])
    cax1 = fig.add_axes([0.75, 0.1, 0.02, 0.25])
    ax2  = fig.add_axes([0.2, 0.4, 0.55, 0.25])
    cax2 = fig.add_axes([0.75, 0.4, 0.02, 0.25])
    ax3  = fig.add_axes([0.2, 0.7, 0.55, 0.25])
    cax3 = fig.add_axes([0.75, 0.7, 0.02, 0.25])
    p1 = griddict['loglike']
    p2 = griddict['chi2']
    p3 = griddict['complexity']
    detected_peaks1 = peakdet2d(p1)
    peakpos1 = np.where(detected_peaks1)
    detected_peaks2 = peakdet2d(p2)
    peakpos2 = np.where(detected_peaks2)
    detected_peaks3 = peakdet2d(p3)
    peakpos3 = np.where(detected_peaks3)
    show_loglike_map(griddict['nu'], np.log10(griddict['tau']), p1, ax1, cax=cax1, 
                     set_contour=True, clevels=None,
                     vmin=-10, vmax=0, 
                     xlabel="$\\nu$", ylabel="$\log\\tau$", zlabel="$\ln \mathcal{L}$",
                     set_normalize=True, peakpos=peakpos1, cmap='jet')
    show_loglike_map(griddict['nu'], np.log10(griddict['tau']), p2, ax2, cax=cax2, 
                     set_contour=True, clevels=None,
                     vmin=-10, vmax=0, 
                     xlabel="$\\nu$", ylabel="$\log\\tau$", zlabel="$\chi^2/2$",
                     set_normalize=True, peakpos=peakpos2, cmap='jet')
    show_loglike_map(griddict['nu'], np.log10(griddict['tau']), p3, ax3, cax=cax3, 
                     set_contour=True, clevels=None,
                     vmin=-10, vmax=0, 
                     xlabel="$\\nu$", ylabel="$\log\\tau$", zlabel="$\mathrm{complexity}$",
                     set_normalize=True, peakpos=peakpos3, cmap='jet')
    plt.show()

def ogle_example_4():
    """ another OGLE example, testing 1d grid optimization using 'varying_tau' but with nu fixed.
    """
    tauarray = np.power(10.0, np.arange(0, 4.2, 0.1))
    truetau  = np.array([80.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
    truenu   = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    fixednu = 1.0
    ttau = 200.0
    tnu = 1.4
    for ttau in truetau:
        for tnu in truenu:
            lcfile = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".dat"
            print("reading %s"%lcfile)
            zydata  = get_data(lcfile)
            record = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_DRW.dat"
            print("writing %s"%record)
            varying_tau(record, zydata, tauarray, fixednu=fixednu, set_verbose=False)
    print("done!")


def ogle_example_5():
    """ testing 2D grid optimization using 'varying_tau_nu' with matern and pareto_exp.
    """
#    tauarray = np.power(10.0, np.arange(0,  4.2, 0.1))
#    nuarray  = np.power(10.0, np.arange(-1, 0.4, 0.1))
    tauarray = [10000,]
#    nuarray  = np.power(10.0, np.linspace(-1, 0.4, 2))
    nuarray  = [2.,]
    ttau = 200.0
    tnu = 1.4
    lcfile = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".dat"
    print("reading %s"%lcfile)
    zydata  = get_data(lcfile)
    record = "dat/OGLE_example/matern_T"+str(ttau)+"_N"+str(tnu)+".test2dgrid.dat"
#    record = "dat/OGLE_example/pow_pareto_T"+str(ttau)+"_N"+str(tnu)+".test2dgrid.dat"
    print("writing %s"%record)
    varying_tau_nu(record, zydata, tauarray, nuarray, covfunc="matern", set_verbose=True)
#    varying_tau_nu(record, zydata, tauarray, nuarray, covfunc="pareto_exp", set_verbose=True)

if __name__ == "__main__":    
    ogle_example_5()




