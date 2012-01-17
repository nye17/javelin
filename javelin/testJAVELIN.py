#Last-modified: 17 Jan 2012 12:34:33 AM
import numpy as np
from predict import Predict

"""
Test from scratch.
"""

def genMockLC(lcfile):
    """ generate mock continuum light curves
    """
    covfunc = "drw"
    tau, sigma = 50.0, 0.5
    lcmean = 10.0
    emean  = lcmean*0.05
    P = Predict(lcmean=lcmean, covfunc=covfunc, tau=tau, sigma=sigma)
    j = np.concatenate([np.arange(0, 140, 1), np.arange(140+180, 2*140+180, 1)])
    ewant  = emean*np.ones_like(j)
    mwant  = P.generate(j, nlc=1, ewant=ewant, errcov=0.0)
    np.savetxt(lcfile, np.vstack((j, mwant, ewant)).T)
    mve, var = P.mve_var(j)
    sig = np.sqrt(var)
    return(mve, sig)


def main():
    lcfile = "mock.dat"
    mve, sig = genMockLC(lcfile)
    pass

if __name__ == "__main__":    
    main()
