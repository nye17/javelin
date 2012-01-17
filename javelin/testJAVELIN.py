#Last-modified: 17 Jan 2012 12:16:13 PM
import numpy as np
from predict import Predict
import matplotlib.pyplot as plt


"""
Test from scratch.
"""

def genMockLC(j, tau, sigma, lcmean, emean):
    """ generate mock continuum light curves
    """
    covfunc = "drw"
    P = Predict(lcmean=lcmean, covfunc=covfunc, tau=tau, sigma=sigma)
    ewant  = emean*np.ones_like(j)
    mwant  = P.generate(j, nlc=1, ewant=ewant, errcov=0.0)
    mve, var = P.mve_var(j)
    sig = np.sqrt(var)
    return(mwant, ewant, mve, sig)

def showSnake(ax, j, mwant, ewant, mve, sig):
    x = np.concatenate((j, j[::-1]))
    y = np.concatenate((mve-sig, (mve+sig)[::-1]))
    ax.fill(x,y,facecolor='.8',edgecolor='1.')
    ax.plot(j, mve, 'k-.')
    ax.errorbar(j, mwant, yerr=ewant, ecolor='k', ms=6, mfc='r', mec='k',
            ls='None')
    ax.set_xlim(j[0], j[-1])

def main():
    if False :
        lcfile = "mock_realistic.dat"
        j = np.concatenate([np.arange(0, 140, 1), np.arange(140+180, 2*140+180, 1)])
        tau, sigma, lcmean = 50.0, 0.5, 10.0
        emean = lcmean*0.05
        mwant, ewant, mve, sig = genMockLC(j, tau, sigma, lcmean, emean)
        np.savetxt(lcfile, np.vstack((j, mwant, ewant)).T)
    if True :
        lcfile = "mock_dense.dat"
        j = np.arange(0, 100, 0.01)
        tau, sigma, lcmean = 50.0, 0.5, 10.0
        emean = lcmean*0.05
        mwant, ewant, mve, sig = genMockLC(j, tau, sigma, lcmean, emean)
        np.savetxt(lcfile, np.vstack((j, mwant, ewant)).T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    showSnake(ax, j, mwant, ewant, mve, sig)
    plt.show()

    fig.clf()
    pass

if __name__ == "__main__":    
    main()
