#Last-modified: 21 Nov 2011 09:11:26 PM


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#from pyspear.gp import *
#from pyspear.gp.cov_funs import matern
from prh import SimpleCovariance1D, covfunc_dict





def plotcov(ax, covfunc="pow_exp", color="k", ls="-", lw=1, alpha=1.0,
        scale=1.0, **par3rd):
    """ demo plot for various covariances
    with both dt/tau and sigma fixed to be one.
    """
    x=np.arange(0.,5.,.01)
    if covfunc in covfunc_dict:
        cf = covfunc_dict[covfunc]
        C = SimpleCovariance1D(eval_fun=cf, amp=1.0, scale=scale, **par3rd)
        y = C(x,0)
        ax.plot(x,y,color=color, ls=ls, lw=lw, alpha=alpha)
    else:
        print("covfuncs currently implemented:")
        print(" ".join(covfunc_dict.keys))
        raise RuntimeError("%s has not been implemented"%covfunc)


def plot_powexp(ax):
    covfunc = "pow_exp"
    nuarr= np.arange(0.1, 1.9, 0.1)
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                ls="-", pow=nu)

def plot_matern(ax):
    covfunc = "matern"
    nuarr= np.power(10.0, np.arange(-1, 1, 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=np.sqrt(2.0), ls="-", diff_degree=nu)

def plot_matern(ax):
    covfunc = "quadratic"
    nuarr= np.power(10.0, np.arange(-1, 1, 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=np.sqrt(2.0), ls="-", diff_degree=nu)

def plot_drw(ax):
    covfunc = "pow_exp"
    print("plot %s "%"drw")
    plotcov(ax, covfunc=covfunc, color="k", ls="-", lw=2, pow=1.0)
    ax.axvline(1.0, color="k", ls="--")

def main(set_log=False):
    fig = plt.figure(figsize=(10,4))
    ax1  = fig.add_axes((0.1,  0.15, 0.35, 0.8))
    ax2  = fig.add_axes((0.55, 0.15, 0.35, 0.8))
    plot_powexp(ax1)
    plot_drw(ax1)
    plot_matern(ax2)
    plot_drw(ax2)
    ax1.set_xlabel("$\Delta t/\\tau_d$")
    ax1.set_ylabel("$\\xi$")
    ax2.set_xlabel("$\Delta t/\\tau_d$")
    ax2.set_ylabel("$\\xi$")
    ax1.set_ylim(1e-3, 1)
    ax2.set_ylim(1e-3, 1)
    if set_log:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    plt.show()

if __name__ == "__main__":    
#    main(set_log=True)
    main(set_log=False)
