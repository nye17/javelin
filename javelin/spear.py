#Last-modified: 01 Mar 2012 01:06:05 AM

#from javelin.spear_covfunc import spear_covfunc as SCF
from spear_covfunc import spear_covfunc as SCF
import numpy as np

from javelin.threadpool import get_threadpool_size, map_noreturn
from javelin.gp import isotropic_cov_funs 
from javelin.gp.GPutils import regularize_array


""" The SPEAR covariance function, called by gp.Covariance.
"""


def spear(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None) :
    if (sigma<0. or tau<0.) :
        raise ValueError, 'The amp and scale parameters must be positive.'

    if (symm is None) :
        symm = (x is y) and (idx is idy)

    x = regularize_array(x)
    y = regularize_array(y)

    nx = x.shape[0]
    ny = y.shape[0]

    if np.isscalar(idx) :
        idx = np.ones(nx, dtype="int", order="F")*idx
    if np.isscalar(idy) :
        idy = np.ones(nx, dtype="int", order="F")*idy

    # Figure out how to divide job up between threads (along y)
    n_threads = min(get_threadpool_size(), nx*ny / 10000)
    
    if n_threads > 1 :
        if not symm:
            # divide ny evenly if x is not y
            bounds = np.linspace(0,ny,n_threads+1)
        else :
            # divide ny*ny evenly in quadrature if x is y
            bounds = np.array(np.sqrt(np.linspace(0,ny*ny,n_threads+1)),dtype=int)

    # Allocate the matrix
    C = np.asmatrix(np.empty((nx,ny),dtype=float,order='F'))

    def targ(C,x,y,idx,idy,cmin,cmax,symm) :
        SCF.covmat_bit(C,x,y,idx,idy,sigma,tau,lags,wids,scales,cmin,cmax,symm)

    if n_threads <= 1 :
        targ(C,x,y,idx,idy,0,-1,symm)
    else :
        thread_args = [(C,x,y,idx,idy,bounds[i],bounds[i+1],symm) for i in xrange(n_threads)]
        map_noreturn(targ, thread_args)

    if symm:
        isotropic_cov_funs.symmetrize(C)

    return(C)


if __name__ == "__main__":    
    npt = 1000
    x   = np.arange(0, npt, 1)
    idx = np.ones(npt,dtype="int")
#    idx[2] = 2
    y   = x
    idy = idx

    sigma = 1.
    tau   = 1.
    lags  = np.array([0.])
    wids  = np.array([0.])
    scales= np.array([1.])
#    lags  = np.array([0., 2.])
#    wids  = np.array([0., 2.])
#    scales= np.array([1., 2.])
#    C = spear_w(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None)
    C = spear(x,y,1,1,sigma,tau,lags,wids,scales,symm=None)
    print(C[:3,:3])
#    print(C)
