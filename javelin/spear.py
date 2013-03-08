#Last-modified: 08 Mar 2013 04:26:31 PM

#from javelin.spear_covfunc import spear_covfunc as SCF
from spear_covfunc import spear_covfunc as SCF
import numpy as np

from javelin.threadpool import get_threadpool_size, map_noreturn
from javelin.gp import isotropic_cov_funs 
from javelin.gp.GPutils import regularize_array

import unittest

""" The SPEAR covariance function, wrapper for the Fortran version.
"""


def spear_threading(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None,set_pmap=False,blocksize=10000) :
    """
    threaded version, divide matrix into subblocks with *blocksize* 
    elements each. Do not use it when multiprocessing is on (e.g., in emcee MCMC
    sampling).

    set_pmap needs to be turned on for the Pmap_Model.
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
    if set_pmap :
        def targ(C,x,y,idx,idy,cmin,cmax,symm) :
            SCF.covmatpmap_bit(C,x,y,idx,idy,sigma,tau,lags,wids,scales,cmin,cmax,symm)
    else :
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


def spear(x,y,idx,idy,sigma,tau,lags,wids,scales,symm=None,set_pmap=False) :
    """
    Clean version without multithreading. Used when multiprocessing is on (e.g., 
    in emcee MCMC sampling).

    set_pmap needs to be turned on for the Pmap_Model.
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
    if set_pmap :
        SCF.covmatpmap_bit(C,x,y,idx,idy,sigma,tau,lags,wids,scales,0,-1,symm)
    else :
        SCF.covmat_bit(C,x,y,idx,idy,sigma,tau,lags,wids,scales,0,-1,symm)
    if symm:
        isotropic_cov_funs.symmetrize(C)
    return(C)


class PmapCovTest(unittest.TestCase):
    def testPmapCov(self):
        # fixed
        jdarr = np.array([0, 1, 0, 1]) # not sorted here.
        idarr = np.array([1, 1, 2, 2])
        # parameters, can be changed
        tau   = 1.0
        sigma = 1.0
        lags  = np.array([0.00, 0.25, 0.00])
        wids  = np.array([0.00, 0.00, 0.00])
        scales= np.array([1.00, 3.00, 2.00])
        C_true= np.empty((4,4), order="F")
        # diagonal
        C_true[0, 0] = 1.0
        C_true[1, 1] = 1.0
        C_true[2, 2] = scales[2]**2 + scales[2]*scales[1]*np.exp(-lags[1]/tau) + scales[1]*scales[2]*np.exp(-lags[1]/tau) + scales[1]**2
        C_true[3, 3] = scales[2]**2 + scales[2]*scales[1]*np.exp(-lags[1]/tau) + scales[1]*scales[2]*np.exp(-lags[1]/tau) + scales[1]**2 
        # off
        C_true[0, 1] = C_true[1, 0] = np.exp(-1/tau)
        C_true[0, 2] = C_true[2, 0] = scales[2] + scales[1]*np.exp(-lags[1]/tau)
        C_true[0, 3] = C_true[3, 0] = scales[2]*np.exp(-1/tau) + scales[1]*np.exp(-(1.0-lags[1])/tau)
        C_true[1, 2] = C_true[2, 1] = scales[2]*np.exp(-1/tau) + scales[1]*np.exp(-(1.0+lags[1])/tau)
        C_true[1, 3] = C_true[3, 1] = scales[2] + scales[1]*np.exp(-lags[1]/tau)
        C_true[2, 3] = C_true[3, 2] = scales[2]*scales[2]*np.exp(-1/tau) + scales[2]*scales[1]*np.exp(-(1.0-lags[1])/tau) + scales[2]*scales[1]*np.exp(-(1.0+lags[1])/tau) + scales[1]*scales[1]*np.exp(-1/tau)
        C_true = C_true * sigma * sigma
        print "Truth :"
        print C_true
        # calculate from spear
        C_thread = spear_threading(jdarr, jdarr, idarr, idarr, sigma, tau, lags, wids, scales, symm=None, set_pmap=True)
        print "Calculated (threading) :"
        print C_thread
        C_bare = spear(jdarr, jdarr, idarr, idarr, sigma, tau, lags, wids, scales, symm=None, set_pmap=True)
        print "Calculated (no threading) :"
        print C_bare
        # compare
        self.assertTrue(np.allclose(C_true, C_thread, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(C_true, C_bare,   rtol=1e-05, atol=1e-08))

if __name__ == "__main__":
    unittest.main()   


