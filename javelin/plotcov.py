#Last-modified: 18 Jan 2012 01:10:35 AM


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from gp import Covariance
from cov import get_covfunc_dict


def plotcov(ax, covfunc="pow_exp", color="k", ls="-", lw=1,
        scale=1.0, amp=1.0, label=None, transparency=1.0, xtuple=None, **par3rd):
    """ demo plot for various covariances
    with both dt/tau and sigma fixed to be one.
    """
    if xtuple is None :
        x=np.arange(0.,5.,.01)
    else :
        x=np.arange(xtuple[0], xtuple[1], xtuple[2])
    covfunc_dict = get_covfunc_dict(covfunc=covfunc, tau=scale, sigma=amp, **par3rd)
    C = Covariance(**covfunc_dict)
    y = C(x,0)
    ax.plot(x,y,color=color, ls=ls, lw=lw, alpha=transparency, label=label)


def plot_powexp(ax):
    covfunc = "pow_exp"
    nuarr= np.arange(0.1, 1.9, 0.1)
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                ls="-", nu=nu)

def plot_matern(ax):
    covfunc = "matern"
    nuarr= np.power(10.0, np.arange(-1, np.log10(2.5), 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=np.sqrt(2.0), ls="-", nu=nu)

def plot_paretoexp(ax):
    covfunc = "pareto_exp"
    nuarr= np.power(10.0, np.arange(np.log10(1), np.log10(5), 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for alpha %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", nu=nu)

def plot_powtail(ax):
    covfunc = "pow_tail"
    nuarr= np.arange(0, 2, 0.1)
    for i, nu in enumerate(nuarr):
        print("plot %s for beta %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", nu=nu)

def plot_keplerexp(ax):
    covfunc = "kepler_exp"
    nuarr= np.arange(0.0, 0.9, 0.05)
    for i, nu in enumerate(nuarr):
        print("plot %s for tcut %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", nu=nu)

def plot_drw(ax):
    covfunc = "pow_exp"
    print("plot %s "%"drw")
    plotcov(ax, covfunc=covfunc, color="k", ls="--", lw=2, nu=1.0)
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
#    plot_paretoexp(ax3)
    plot_keplerexp(ax3)
    plot_drw(ax3)
#    ax3.text(0.9, 0.9,'pareto exp',  ha='right', va='top', transform = ax3.transAxes)
    ax3.text(0.9, 0.9,'kepler exp',  ha='right', va='top', transform = ax3.transAxes)
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_xlabel("$\Delta t/\\tau_d$")
        ax.set_ylabel("$\\xi$")
        ax.set_xlim(0, 4.9)
        ax.set_ylim(1e-3, 1.05)
        if set_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":    
    main(set_log=False)
