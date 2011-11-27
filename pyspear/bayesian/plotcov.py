#Last-modified: 27 Nov 2011 12:25:51 AM


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#from pyspear.gp import *
#from pyspear.gp.cov_funs import matern
from prh import SimpleCovariance1D, covfunc_dict





def plotcov(ax, covfunc="pow_exp", color="k", ls="-", lw=1,
        scale=1.0, **par3rd):
    """ demo plot for various covariances
    with both dt/tau and sigma fixed to be one.
    """
    x=np.arange(0.,5.,.01)
    if covfunc in covfunc_dict:
        cf = covfunc_dict[covfunc]
        C = SimpleCovariance1D(eval_fun=cf, amp=1.0, scale=scale, **par3rd)
        y = C(x,0)
        ax.plot(x,y,color=color, ls=ls, lw=lw, alpha=0.8)
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
#    nuarr= np.power(10.0, np.arange(-1, 1, 0.1))
#    nuarr= np.power(10.0, np.arange(-1, 0.4, 0.1))
    nuarr= np.power(10.0, np.arange(-1, np.log10(2), 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=np.sqrt(2.0), ls="-", diff_degree=nu)

def plot_paretoexp(ax):
    covfunc = "pareto_exp"
    nuarr= np.power(10.0, np.arange(-1, 1, 0.1))
    nuarr= np.power(10.0, np.arange(-1, np.log10(5), 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for alpha %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", alpha=nu)

def plot_drw(ax):
    covfunc = "pow_exp"
    print("plot %s "%"drw")
    plotcov(ax, covfunc=covfunc, color="k", ls="--", lw=2, pow=1.0)
    ax.axvline(1.0, color="k", ls="--")

def main(set_log=False):
    fig = plt.figure(figsize=(10,4))
    ax1  = fig.add_axes((0.08, 0.15, 0.30, 0.8))
    ax2  = fig.add_axes((0.38, 0.15, 0.30, 0.8))
    ax3  = fig.add_axes((0.68, 0.15, 0.30, 0.8))
    plot_powexp(ax1)
    plot_drw(ax1)
    ax1.text(0.9, 0.9,'powered exp', ha='right', va='top', transform = ax1.transAxes)
    plot_matern(ax2)
    plot_drw(ax2)
    ax2.text(0.9, 0.9,'matern',      ha='right', va='top', transform = ax2.transAxes)
    plot_paretoexp(ax3)
    plot_drw(ax3)
    ax3.text(0.9, 0.9,'pareto exp',  ha='right', va='top', transform = ax3.transAxes)
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_xlabel("$\Delta t/\\tau_d$")
        ax.set_ylabel("$\\xi$")
        ax.set_xlim(0, 4.9)
        ax.set_ylim(1e-3, 1)
        if set_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":    
#    main(set_log=True)
    main(set_log=False)
