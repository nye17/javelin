#Last-modified: 08 Dec 2013 03:41:24

from __future__ import absolute_import
from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
import numpy as np
from javelin.gp import Covariance
from javelin.cov import get_covfunc_dict
from six.moves import zip


# for citing covfunc properties in plotting
labeldict = {'pow_exp'    : {'nuname':r"$\gamma$",
                             'cfname':r"$\mathrm{powered\,exp}$"},
             'matern'     : {'nuname':r"$\nu$",
                             'cfname':r"$\mathrm{Mat}\acute{\mathrm{e}}\mathrm{rn}$"},
             'pareto_exp' : {'nuname':r"$\beta$",
                             'cfname':r"$\mathrm{Pareto\,exp}$"},
             'kepler_exp' : {'nuname':r"$\tau_\mathrm{cut}/\tau$",
                             'cfname':r"$\mathrm{Kepler\,exp}$"},
             }

# legend font size
prop = font_manager.FontProperties(size=14)


def plotcov(ax, covfunc="pow_exp", color="k", ls="-", lw=1, scale=1.0, amp=1.0, label=None, transparency=1.0, xtuple=None, **par3rd):
    """ demo plot for various covariances
    with both dt/tau and sigma fixed to be one.
    """
    if xtuple is None :
        x=np.arange(0.,5.,.01)
    else :
        x=np.arange(xtuple[0], xtuple[1], xtuple[2])
    covfunc_dict = get_covfunc_dict(covfunc=covfunc, tau=scale, sigma=amp, **par3rd)
    # print covfunc_dict
    C = Covariance(**covfunc_dict)
    y = C(x,0)
    y = np.empty_like(x)
    # this is trying to change a (500,1) array to (500,)
    y[:] = C(x,0).T[0,:]
    ax.plot(x,y,color=color, ls=ls, lw=lw, alpha=transparency, label=label)

def show_nurange(cax, numin, numax, vmin=None, vmax=None, covfunc="pow_exp", set_color=True):
    if set_color:
        cmap = "jet"
    else:
        cmap = "binary" 
    nuname = labeldict[covfunc]["nuname"]
    cfname = labeldict[covfunc]["cfname"]
    if vmin is None: vmin = numin
    if vmax is None: vmax = numax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb   = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
            orientation='horizontal')
    cb.set_ticks([vmin,0.5*(vmin+vmax), vmax])
    cb.set_ticklabels(["$"+str(numin)+"$", "$\\longrightarrow$", "$"+str(numax)+"$"])
    cb.set_label(nuname)
    cax.text(.5, 1.15, cfname, ha='center', va="bottom", fontsize=18)
#    plt.setp(cax.get_xticklabels(), fontsize=12)

def show_drw(ax, xtuple=None, set_color="color"):
    """ plot drw model.
    """
    covfunc = "drw"
    print(("plotting %s "%covfunc))
    plotcov(ax, covfunc=covfunc, color="k", ls="--", lw=2.5, scale=1.0,
            transparency=0.5, xtuple=xtuple, label="$\mathrm{DRW}$")
    ax.axvline(1.0, color="gray", ls="-", lw=0.5, alpha=1.0)

def show_powexp(ax, set_color="color"):
    """ plot pow_exp model.
    """
    covfunc = "pow_exp"
    numin, numax, nnu = 0.01, 1.99, 10
    nuarr = np.linspace(numin, numax, nnu)
    print(("plotting %s model with %4d numbers of nu from %.4g to %.4g\
            "%(covfunc, nnu, numin, numax)))
    for i, nu in enumerate(nuarr):
        if set_color:
            color= cm.jet(1.*i/len(nuarr))
        else:
            color= cm.binary(1.*(i+2)/len(nuarr))
        plotcov(ax, covfunc=covfunc, color=color, ls="-", nu=nu)

def show_matern(ax, set_color="color"):
    """ plot matern model.
    """
    covfunc = "matern"
    numin, numax, nnu = 0.10, 2.50, 10
    nuarr = np.power(10.0, np.linspace(np.log10(numin), np.log10(numax), nnu))
    print(("plotting %s model with %4d numbers of nu from %.4g to %.4g\
            "%(covfunc, nnu, numin, numax)))
    for i, nu in enumerate(nuarr):
        if set_color:
            color= cm.jet(1.*i/len(nuarr))
        else:
            color= cm.binary(1.*(i+2)/len(nuarr))
        plotcov(ax, covfunc=covfunc, color=color, scale=np.sqrt(2.0), ls="-", nu=nu)

def show_paretoexp(ax, set_color="color"):
    """ plot matern model.
    """
    covfunc = "pareto_exp"
    numin, numax, nnu = 1.0, 5.0, 10
    nuarr = np.power(10.0, np.linspace(np.log10(numin), np.log10(numax), nnu))
    print(("plotting %s model with %4d numbers of nu from %.4g to %.4g\
            "%(covfunc, nnu, numin, numax)))
    for i, nu in enumerate(nuarr):
        if set_color:
            color= cm.jet(1.*i/len(nuarr))
        else:
            color= cm.binary(1.*(i+2)/len(nuarr))
        plotcov(ax, covfunc=covfunc, color=color, scale=1.0, ls="-",
                nu=nu)

def show_keplerexp(ax, xtuple=None, set_color="color"):
    """ plot matern model.
    """
    covfunc = "kepler_exp"
    numin, numax, nnu = 0.0, 0.8, 10
    nuarr = np.linspace(numin, numax, nnu)
    print(("plotting %s model with %4d numbers of nu from %.4g to %.4g\
            "%(covfunc, nnu, numin, numax)))
    for i, nu in enumerate(nuarr):
        if set_color:
            color= cm.jet(1.*i/len(nuarr))
        else:
            color= cm.binary(1.*(i+2)/len(nuarr))
        plotcov(ax, covfunc=covfunc, color=color, scale=1.0, ls="-",
                xtuple=xtuple, nu=nu)


def covdemo(set_color=True):
    """ showcasing three different covariance models we used in the paper and
    compared them to the default DRM model.
    """
    # main axis
    xstart = 0.10
    ystart = 0.10
    xgutter= 0.00
    ygutter= 0.00
    width  = 0.44
    height = 0.44
    cwidth = 0.10
    cheight= 0.02
    cxdiff = 0.26
    cydiff = 0.26
    fig = plt.figure(figsize=(8,8.*height/width))
    #topleft
    ax1  = fig.add_axes((xstart, ystart+ygutter+height, width, height))
    #topright
    ax2  = fig.add_axes((xstart+xgutter+width, ystart+ygutter+height, width, height))
    #bottomleft
    ax3  = fig.add_axes((xstart, ystart, width, height))
    #bottomright
    ax4  = fig.add_axes((xstart+xgutter+width, ystart, width, height))
    # colorbar
    cax1 = fig.add_axes((xstart+cxdiff, ystart+ygutter+height+cydiff, cwidth, cheight))
    cax2 = fig.add_axes((xstart+xgutter+width+cxdiff, ystart+ygutter+height+cydiff, cwidth, cheight))
    cax3 = fig.add_axes((xstart+cxdiff, ystart+cydiff, cwidth, cheight))
    cax4 = fig.add_axes((xstart+xgutter+width+cxdiff, ystart+cydiff, cwidth, cheight))
    # pow_exp
    show_powexp(ax1, set_color=set_color)
    show_drw(ax1, set_color=set_color)
    show_nurange(cax1, 0, 2, covfunc="pow_exp", set_color=set_color)
    # matern
    show_matern(ax2, set_color=set_color)
    show_drw(ax2, set_color=set_color)
    show_nurange(cax2, 0.1, 2.5, vmin=np.log10(0.1), vmax=np.log10(2.5),
            covfunc="matern", set_color=set_color)
    # pareto_exp
    show_paretoexp(ax3, set_color=set_color)
    show_drw(ax3, set_color=set_color)
    show_nurange(cax3, 1.0, 5.0, vmin=np.log10(1.0), vmax=np.log10(5.0),
            covfunc="pareto_exp", set_color=set_color)
    # kepler_exp
    show_keplerexp(ax4, set_color=set_color)
    show_drw(ax4, set_color=set_color)
    show_nurange(cax4, 0.0, 0.8,
            covfunc="kepler_exp", set_color=set_color)
    # manipulate labels
    axes = [ax1, ax2, ax3, ax4]
    aids = [r"$a$", r"$b$", r"$c$", r"$d$"]
    for ax, id in zip(axes, aids):
        ax.set_xlim(0, 4.9)
        ax.set_ylim(1e-3, 1)
        ax.legend(loc=1,
                bbox_to_anchor=(1.0, 1.0),
                fancybox=False, frameon=False, shadow=False, prop=prop)
        ax.annotate(id, xy=(0.05,0.05),
            xycoords='axes fraction',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8),
            fontsize=14,)
    ax3.set_xlabel("$\Delta t/\\tau$")
    ax4.set_xlabel("$\Delta t/\\tau$")
    ax1.set_ylabel("$\mathrm{covariance}$")
    ax3.set_ylabel("$\mathrm{covariance}$")
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])
    plt.show()



if __name__ == "__main__":    
    covdemo()
