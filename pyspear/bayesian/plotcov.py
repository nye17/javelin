#Last-modified: 11 Jan 2012 07:51:58 PM


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#from pyspear.gp import *
#from pyspear.gp.cov_funs import matern
from prh import SimpleCovariance1D, covfunc_dict





def plotcov(ax, covfunc="pow_exp", color="k", ls="-", lw=1,
        scale=1.0, label=None, transparency=1.0, **par3rd):
    """ demo plot for various covariances
    with both dt/tau and sigma fixed to be one.
    """
    x=np.arange(0.,5.,.001)
    if covfunc in covfunc_dict:
        cf = covfunc_dict[covfunc]
        C = SimpleCovariance1D(eval_fun=cf, amp=1.0, scale=scale, **par3rd)
        y = C(x,0)
        ax.plot(x,y,color=color, ls=ls, lw=lw, alpha=transparency, label=label)
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
    nuarr= np.power(10.0, np.arange(-1, np.log10(2.5), 0.1))
    for i, nu in enumerate(nuarr):
        print("plot %s for nu %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=np.sqrt(2.0), ls="-", diff_degree=nu)

def plot_paretoexp(ax):
    covfunc = "pareto_exp"
#    nuarr= np.power(10.0, np.arange(-1, 1, 0.1))
#    nuarr= np.power(10.0, np.arange(-1, np.log10(5), 0.1))
#    nuarr= np.power(10.0, np.arange(np.log10(1), np.log10(5), 0.1))
    nuarr= np.array([1.5, 2.0, 2.5, 3.0])
    for i, nu in enumerate(nuarr):
        print("plot %s for alpha %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", alpha=nu)

def plot_powtail(ax):
    covfunc = "pow_tail"
    nuarr= np.arange(0, 2, 0.1)
    for i, nu in enumerate(nuarr):
        print("plot %s for beta %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", beta=nu)

def plot_keplerexp(ax):
    covfunc = "kepler_exp"
    nuarr= np.arange(0.0, 0.5, 0.05)
    for i, nu in enumerate(nuarr):
        print("plot %s for tcut %10.5f"%(covfunc, nu))
        plotcov(ax, covfunc=covfunc, color=cm.jet(1.*i/len(nuarr)), 
                scale=1., ls="-", tcut=nu)

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
#    plot_paretoexp(ax3)
#    plot_keplerexp(ax3)
    plot_powtail(ax3)
    plot_drw(ax3)
#    ax3.text(0.9, 0.9,'pareto exp',  ha='right', va='top', transform = ax3.transAxes)
#    ax3.text(0.9, 0.9,'kepler exp',  ha='right', va='top', transform = ax3.transAxes)
    ax3.text(0.9, 0.9,'powered tail',  ha='right', va='top', transform = ax3.transAxes)
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

def single(set_log=False, prop={"pow_exp":   (1.0, 1.000, 1.0), 
                                "matern" :   (1.0, 1.414, 0.5),
                                "pareto_exp":(1.0, 1.000, 0.5)},
                          xmax=None):
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    if xmax is None:
        xmax = 0
        for par in prop.itervalues():
            tau = par[1]
            if tau > xmax:
                xmax = tau
        xmax = xmax*3
        if xmax > 2000.0:
            xmax = 2000.0
    x = np.linspace(0.0, xmax, 100)
    for covfunc in prop.iterkeys():
        v=prop[covfunc]
        print(covfunc)
        if covfunc == "drw": 
            cf = covfunc_dict[covfunc]
            C = SimpleCovariance1D(eval_fun=cf, amp=v[0], scale=v[1], pow=1.0)
            y = C(x,0)
            ax.plot(x,y, ls="-", color="gray", lw=4, alpha=0.1, label=covfunc)
            ax.axvline(v[1], color="gray",  ls="-", lw=4, alpha=0.1)
        elif covfunc == "pow_exp": 
            cf = covfunc_dict[covfunc]
            C = SimpleCovariance1D(eval_fun=cf, amp=v[0], scale=v[1], pow=v[2])
            y = C(x,0)
            ax.plot(x,y, ls="-", color="k", lw=3, alpha=0.3, label=covfunc)
            ax.axvline(v[1], color="k", lw=3,  ls="-", alpha=0.3)
        elif covfunc == "matern":
            cf = covfunc_dict[covfunc]
            C = SimpleCovariance1D(eval_fun=cf, amp=v[0], scale=v[1], diff_degree=v[2])
            y = C(x,0)
            ax.plot(x,y, ls="--", color="g", lw=2, alpha=0.6, label=covfunc)
            ax.axvline(v[1], color="g", lw=2, ls="--", alpha=0.6)
        elif covfunc == "pareto_exp":
            cf = covfunc_dict[covfunc]
            C = SimpleCovariance1D(eval_fun=cf, amp=v[0], scale=v[1], alpha=v[2])
            y = C(x,0)
            ax.plot(x,y, ls=":", color="r", lw=1, alpha=0.9, label=covfunc)
            ax.axvline(v[1], color="r", lw=1, ls=":", alpha=0.9)
        else:
            raise RuntimeError("no %s found"%covfunc)
        ax.legend(loc=1)
    plt.show()


if __name__ == "__main__":    
    main(set_log=False)
#    single(set_log=False)
    prop={
#          "drw":       (0.085,   409.4915,    1.0000), 
#          "pow_exp":   (0.0867,  494.1713,    0.9200), 
#          "matern" :   (0.0858,  719.6857,    0.4319),
#          "pareto_exp":(0.0847,  494.1713,    0.5649),
#          "pareto_exp":(0.0847,  494.1713,    1.5),
#          "pareto_exp":(0.0732,  281.1769,    3.2374),

#          "pareto_exp":(0.085,   409.4915,    0.8875),
#          "pareto_exp":(0.0847,  409.4915,    0.5469 ),
# smc105.5_i_32347
#          "drw":       (0.1017,  494.1713,    1.0000 ), 
#          "pow_exp":   (0.1194,  596.3623,    1.0600 ), 
#          "matern" :   (0.1153,  719.6857,    0.5603 ),
#          "pareto_exp":(0.1362,  868.5114,    5.0000 ),
# sandbox
          "drw":       (1.0,     10.,      1.0000 ), 
#          "pow_exp":   (0.9981,  8.685,    1.053  ), 
          "pareto_exp":(1.147,   13.38,    1.648 ),
#          "pareto_exp":(1.036,     10.78,    0.7368),
          "pareto_exp":(2.832,      84.04,   0.0),
         }
#    single(set_log=False, prop=prop, xmax=100)
