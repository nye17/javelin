#Last-modified: 05 Mar 2012 07:29:17 PM

#from javelin.spear_covfunc import spear_covfunc as SCF
from spear_covfunc import spear_covfunc as SCF
import numpy as np

from javelin.threadpool import get_threadpool_size, map_noreturn
from javelin.gp import isotropic_cov_funs 
from javelin.gp.GPutils import regularize_array


""" The SPEAR covariance function, wrapper for the Fortran version.
"""


def spear_threading(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None,
        blocksize=10000) :
    """
    threaded version, divide matrix into subblocks with *blocksize* 
    elements each. Do not use it when multiprocessing is on (e.g., in emcee MCMC
    sampling).
    """
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
    n_threads = min(get_threadpool_size(), nx*ny/blocksize)
    
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


def spear(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None) :
    """
    Clean version without multithreading. Used when multiprocessing is on (e.g., 
    in emcee MCMC sampling).
    """
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

    # Allocate the matrix
    C = np.asmatrix(np.empty((nx,ny),dtype=float,order='F'))

    SCF.covmat_bit(C,x,y,idx,idy,sigma,tau,lags,wids,scales,0,-1,symm)

    if symm:
        isotropic_cov_funs.symmetrize(C)

    return(C)
