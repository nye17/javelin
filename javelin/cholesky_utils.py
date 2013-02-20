#Last-modified: 19 Feb 2013 10:10:06 PM

__all__ = ['cholesky', 'trisolve', 'chosolve', 'chodet', 'chosolve_from_tri', 'chodet_from_tri']

import unittest
from javelin.gp.linalg_utils import dpotrf_wrap, dpotrf2_wrap, dtrsm_wrap
import numpy as np
from numpy.testing import assert_equal,  assert_almost_equal, assert_array_equal


""" Compilation of methods for solving linear systems using Cholesky
decomposition.
"""


def cholesky(A, nugget=None, inplace=False, raiseinfo=True):
    """ Cholesky into upper triangular matrix.

    Note that U is a fortran-array.

    U = cholesky(A, nugget=None])

    """
    n = A.shape[0]
    if inplace:
        U=A
    else:
        U = A.copy('F')
    if nugget is not None:
        for i in xrange(n):
            U[i,i] += nugget[i]
    info = dpotrf_wrap(U)
    if raiseinfo:
        if info>0:
            raise RuntimeError("Matrix does not appear to be positive definite by row %i." % info)
        else:
            return(U)
    else:
        return(U, info)

def cholesky2(A, nugget=None, inplace=False, raiseinfo=True):
    """ Cholesky into lower triangular matrix.
    L = cholesky2(A, nugget=None])

    Note that L is a fortran-array.

    """
    n = A.shape[0]
    if inplace:
        L=A
    else:
        L = A.copy('F')
    if nugget is not None:
        for i in xrange(n):
            L[i,i] += nugget[i]
    info = dpotrf2_wrap(L)
    if raiseinfo:
        if info>0:
            raise RuntimeError("Matrix does not appear to be positive definite by row %i." % info)
        else:
            return(L)
    else:
        return(L, info)

def trisolve(U,b,uplo='U',transa='N',alpha=1.,inplace=False):
    """
    x = trisolve(U,b, uplo='U')

    Solves op(U) x = alpha*b, where U is upper triangular if uplo='U'
    or lower triangular if uplo = 'L'.

    op(U) is U if transa='N', or U^T if transa='T'

    alpha is default to be 1.

    If a degenerate column is found, an error is raised.
    """
    if inplace:
        x = b
    else:
        x = b.copy('F')
    if U.shape[0] == 0:
        raise ValueError, 'Attempted to solve zero-rank triangular system'
    dtrsm_wrap(a=U,b=x,side='L',uplo=uplo,transa=transa,alpha=alpha)
    return(x)

def chosolve_from_tri(U, b, nugget=None, inplace=False):
    """ 
        solve A   x =b given U where A = U^T U, by the following steps:
        solve U^T w =b for w
        solve U   x =w for x = A^-1 b
    """
    w = trisolve(U,b,uplo='U',transa='T',alpha=1.,inplace=inplace)
    x = trisolve(U,w,uplo='U',transa='N',alpha=1.,inplace=inplace)
    return(x)

def chosolve(A, b, nugget=None, inplace=False):
    """ 
        solve A   x =b by the following steps:
        decompose A to be U^T U
        solve U^T w =b for w
        solve U   x =w for x = A^-1 b
    """
    U = cholesky(A, nugget=nugget, inplace=inplace)
    x = chosolve_from_tri(U, b, nugget=nugget, inplace=inplace)
    return(x)

def chodet_from_tri(U, nugget=None, retlog=True):
    """
    det(A) = |A| = product {U_ii^2} for i=[1:n]
    log(det(A)) = log(|A|) = 2 sum {log(U_ii)} for i=[1:n]
    """
    n = U.shape[0]
    if retlog:
        logdet = 0.0
        for i in xrange(n):
            logdet = logdet + np.log(U[i,i])
        logdet = logdet*2.0
        return(logdet)
    else:
        det = 1.0
        for i in xrange(n):
            det = det*U[i,i]**2
        return(det)

def chodet(A, nugget=None, retlog=True):
    """
    det(A) = |A| = product {U_ii^2} for i=[1:n]
    log(det(A)) = log(|A|) = 2 sum {log(U_ii)} for i=[1:n]
    """
    U = cholesky(A, nugget=nugget, inplace=False, raiseinfo=True)
    d = chodet_from_tri(U, nugget=nugget, retlog=retlog)
    return(d)



class CholeskyTests(unittest.TestCase):

    def testCholesky_Decompose(self):
        A = np.array([
                      [25., -5., 10.], 
                      [-5., 17., 10.],
                      [10., 10., 62.],
                     ])
        U = cholesky(A)
        B = np.array([
                      [ 5., -1.,  2.], 
                      [ 0.,  4.,  3.],
                      [ 0.,  0.,  7.],
                     ])
        assert_almost_equal(U, B)
        assert_almost_equal(np.dot(U.T, U), A)

    def testCholesky2_Decompose(self):
        A = np.array([
                      [25., -5., 10.], 
                      [-5., 17., 10.],
                      [10., 10., 62.],
                     ])
        L = cholesky2(A)
        print "L"
        print L
        B = np.array([
                      [ 5.,  0.,  0.], 
                      [-1.,  4.,  0.],
                      [ 2.,  3.,  7.],
                     ])
        assert_almost_equal(L, B)
        assert_almost_equal(np.dot(L, L.T), A)

    def testCholesky_Solve(self):
        b = np.array([55., -19., 114.]).T
#        b = regularize_array(b)
        A = np.array([
                      [25., -5., 10.], 
                      [-5., 17., 10.],
                      [10., 10., 62.],
                     ])
        x = chosolve(A, b)
        a = np.array([1., -2., 2.]).T
        assert_almost_equal(x, a)
        assert_almost_equal(np.dot(A, x), b)

    def testCholesky_Det(self):
        A = np.array([
                      [25., -5., 10.], 
                      [-5., 17., 10.],
                      [10., 10., 62.],
                     ])
        d = 19600.0
        logd = np.log(d)
        t = chodet(A, retlog=False)
        logt = chodet(A, retlog=True)
        self.assertAlmostEqual(d, t)
        self.assertAlmostEqual(logd, logt)
        


if __name__ == "__main__":    
    unittest.main()
