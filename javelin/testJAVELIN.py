#Last-modified: 03 Feb 2012 05:33:32 PM
import numpy as np
from predict import Predict
from psd import psd
from lcio import *
from zylc import zyLC, get_data
import matplotlib.pyplot as plt

"""
Test from scratch.
"""
# plotting
def plotSnake(ax, j, mve, sig):
    """ plot error snakes.
    """
    x = np.concatenate((j, j[::-1]))
    y = np.concatenate((mve-sig, (mve+sig)[::-1]))
    ax.fill(x,y,facecolor='.8',edgecolor='1.')
    ax.plot(j, mve, 'k--')

def plotLC(ax, j, m, e, mfc="r"):
    """ dotted out light curves
    """
    ax.errorbar(j, m, yerr=e, ecolor='k', marker="o", ms=4, mfc=mfc, mec='k', ls='None')

# computing
def genMockLC(jwant, lcmean, emean, covfunc='drw', **covparams):
    """ generate mock continuum light curves.
    """
    P = Predict(lcmean=lcmean, covfunc=covfunc, **covparams)
    ewant  = emean*np.ones_like(jwant)
    mwant  = P.generate(jwant, nlc=1, ewant=ewant, errcov=0.0)
    return(mwant, ewant)

def genContinuum(tau=400.0, sigma=0.5, lcmean=10.0, frac_err=0.05):
    """ generate a DRW continuum light curve
    """
    # an example of observation epochs, five seasons with weekly cadence
    jwant = np.concatenate([np.arange(0,               160,         6), 
                            np.arange(160+180,         2*160+180,   6),
                            np.arange(2*160+2*180,     3*160+2*180, 6),
                            np.arange(3*160+3*180,     4*160+3*180, 6),
                            np.arange(4*160+4*180,     5*160+4*180, 6),
                            np.arange(5*160+5*180,     6*160+5*180, 6),
                            ])
    emean = lcmean*frac_err
    mwant, ewant = genMockLC(jwant, lcmean, emean, covfunc="drw", tau=tau, sigma=sigma)
    return(jwant, mwant, ewant)


def file_exists(fname) :
    try :
        f = open(fname, "r")
        f.close()
        return(True)
    except :
        return(False)


def main(set_plot=True):
    lccon = "dat/loopdeloop_con.dat"
    if file_exists(lccon) :
        print("mock continuum file %s exists"%lccon)
    else :
        print("generate mock continuum file %s"%lccon)
        j, m, e = genContinuum(tau=100.0, sigma=0.5, lcmean=10.0, frac_err=0.05)
        lclist = [[j, m, e]]
        writelc(lclist, lccon)

    # Read the continuum file into a zyLC object
    datacon = get_data(lccon, names=["continuum"], set_subtractmean=False)
    if set_plot :
        datacon.plot()

    # Running MCMC
    #FIXME


if __name__ == "__main__":    
    main()
