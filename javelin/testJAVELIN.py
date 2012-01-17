#Last-modified: 17 Jan 2012 04:43:33 PM
import numpy as np
from predict import Predict
from psd import psd
import matplotlib.pyplot as plt


"""
Test from scratch.
"""

def genMockLC(j, lcmean, emean, covfunc='drw', **covparams):
    """ generate mock continuum light curves
    """
    P = Predict(lcmean=lcmean, covfunc=covfunc, **covparams)
    ewant  = emean*np.ones_like(j)
    mwant  = P.generate(j, nlc=1, ewant=ewant, errcov=0.0)
    mve, var = P.mve_var(j)
    sig = np.sqrt(var)
    return(mwant, ewant, mve, sig)

def showSnake(ax, j, mwant, ewant, mve, sig, mfc="r"):
    x = np.concatenate((j, j[::-1]))
    y = np.concatenate((mve-sig, (mve+sig)[::-1]))
    ax.fill(x,y,facecolor='.8',edgecolor='1.')
    ax.plot(j, mve, 'k--')
    ax.errorbar(j, mwant, yerr=ewant, ecolor='k', marker="o", ms=4, mfc=mfc, mec='k', ls='None')
    ax.set_xlim(j[0], j[-1])

def main():
    lcfile1 = "mock_realistic.dat"
    if False :
        j = np.concatenate([np.arange(0, 140, 1), np.arange(140+180, 2*140+180, 1)])
        tau, sigma, lcmean = 50.0, 0.5, 10.0
        emean = lcmean*0.05
        mwant, ewant, mve, sig = genMockLC(j, lcmean, emean, covfunc="drw",
                tau=tau, sigma=sigma)
        np.savetxt(lcfile1, np.vstack((j, mwant, ewant)).T)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        showSnake(ax, j, mwant, ewant, mve, sig)
        plt.show()

    lcfile2 = "mock_dense_drw.dat"
    lcfile3 = "mock_dense_kpl.dat"
    dj = 0.01
    if False :
        j = np.arange(0, dj*1024*8, dj)
        tau, sigma, lcmean = 5.0, 0.5, 10.0
        nu = 0.05
        emean = lcmean*0.00
        mwant2, ewant2, mve2, sig2 = genMockLC(j, lcmean, emean,
                covfunc="kepler_exp",
                tau=tau, sigma=sigma, nu=nu)
        np.savetxt(lcfile2, np.vstack((j, mwant2, ewant2)).T)
        #
        mwant3, ewant3, mve3, sig3 = genMockLC(j, lcmean, emean,
                covfunc="drw",
                tau=tau, sigma=sigma)
        np.savetxt(lcfile3, np.vstack((j, mwant3, ewant3)).T)
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        showSnake(ax, j, mwant2, ewant2, mve2, sig2, mfc="r")
        showSnake(ax, j, mwant3, ewant3, mve3, sig3, mfc="g")
        plt.show()

    if True :
        j2, m2 = np.genfromtxt(lcfile2, usecols=(0,1), unpack=True)
        p2, f2 = psd(m2, NFFT=len(j2)//2, Fs=1/dj)
        j3, m3 = np.genfromtxt(lcfile3, usecols=(0,1), unpack=True)
        p3, f3 = psd(m3, NFFT=len(j3)//2, Fs=1/dj)
        fig = plt.figure(3)
        ax = fig.add_subplot(111)
        ax.plot(f2, p2, "r.", alpha=0.8)
        ax.plot(f3, p3*1000, "g.", alpha=0.4)
        nunit = np.searchsorted(f2, 1.)
        ax.plot(f2, f2**-2*p2[nunit], "k--", alpha=0.5)
        ax.plot(f2, f2**-3*p2[nunit], "k--", alpha=0.5)
        nunit = np.searchsorted(f3, 1.)
        ax.plot(f3, f3**-2*p3[nunit]*1000, "k--", alpha=0.5)
        ax.plot(f3, f3**-3*p3[nunit]*1000, "k--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.show()


    pass

if __name__ == "__main__":    
    main()
