#Last-modified: 25 Oct 2011 03:20:57 AM

from pyspear.gp import Mean, Covariance, observe, Realization, GPutils, FullRankCovariance
from pyspear.gp.cov_funs import matern, quadratic, gaussian, pow_exp, sphere
from pyspear.zylc import zyLC

from cholesky_utils import cholesky, trisolve, chosolve, chodet, chosolve_from_tri, chodet_from_tri

import numpy as np
from numpy.random import normal, multivariate_normal

covfunc_dict = {
                "matern"    :  matern.euclidean,
                "pow_exp"   :  pow_exp.euclidean,
                "drw"       :  pow_exp.euclidean,
                "gaussian"  :  gaussian.euclidean,
#                "quadratic" : quadratic.euclidean,
#                "sphere"    :    sphere.euclidean,
               }

class PRH(object):
    """
    """
    def __init__(self, zylc,
                       covfunc="pow_exp", 
                       tau=20.0, sigma=0.5, nu=1,
                       ):

        # get self.zylc, attributes[[jme]list, [jmei]arr]
        if not isinstance(zylc, zyLC):
            raise RuntimeError("zylc has to be a zyLC object")
        else:
            self.zylc = zylc
            self.npt  = zylc.npt
            self.nlc  = zylc.nlc
            self.jarr = zylc.jarr
            self.marr = zylc.marr.T  # make it a vector
            self.earr = zylc.earr
            self.iarr = zylc.iarr
            self.varr = np.power(zylc.earr, 2)
            self.larr = np.zeros((self.npt, self.nlc))
            for i in xrange(self.npt):
                lcid = self.iarr[i] - 1
                self.larr[i, lcid] = 1.0
            self.larrTr = self.larr.T

        if (self.nlc == 1):
            print("Single light curve modeling, using covariance from gp module.")
            self.set_single = True
        else:
            print("Multiple light curve modeling, using only DRW model.")
            self.set_single = False

        if self.set_single:
            # get self.C as Covariance, methods[cholesky, continue_cholesky, observe]
            if covfunc in covfunc_dict:
                cf = covfunc_dict[covfunc]
                if covfunc == "matern":
                    self.C  = FullRankCovariance(eval_fun = cf, amp=sigma, scale=tau, 
                            diff_degree=nu)
                elif covfunc == "pow_exp":
                    self.C  = FullRankCovariance(eval_fun = cf, amp=sigma, scale=tau, 
                            pow=nu)
                elif covfunc == "drw":
                    self.C  = FullRankCovariance(eval_fun = cf, amp=sigma, scale=tau, 
                            pow=1.0)
                else:
                    self.C  = FullRankCovariance(eval_fun = cf, amp=sigma, scale=tau)
            else:
                print("covfuncs currently implemented:")
                print(" ".join(covfunc_dict.keys))
                raise RuntimeError("%s has not been implemented"%covfunc)

            output = self.loglike_prh(self.set_single)
            print(output)
#            U, smatrix = self.C.cholesky(self.jarr, nugget=self.varr, return_eval_also=True)
#            output = self.loglike_prh(False, U=U)
#            print(output)
        else:
            raise RuntimeError("Sorry, not implemented yet")


    def loglike_prh(self, set_single, U=None, retq=True):
        # cholesky decompose S+N so that U^T U = S+N = C
        if set_single:
            U, smatrx = self.C.cholesky(self.jarr, nugget=self.varr, return_eval_also=True)
        elif U is None:
            raise RuntimeError("require U of cpnmatrix for more RM purposes")

        detC_log = chodet_from_tri(U, retlog=True)
        # solve for C a = y so that a = C^-1 y 
        a = chosolve_from_tri(U, self.marr)
        # solve for C b = L so that b = C^-1 L
        b = chosolve_from_tri(U, self.larr)
        # multiply L^T and b so that C_p = L^T C^-1 L = C_q^-1
        C_p = np.dot(self.larrTr, b)

        # for 'set_single is True' case, C_p is a scalar.
        if set_single:
            # for single-mode, cholesky of C_p is simply squre-root of C_p
            W = np.sqrt(C_p)
            detCp_log = np.log(C_p.squeeze())
            # for single-mode, simply devide L^T by C_p
            d = self.larrTr/C_p
        else:
            # cholesky decompose C_p so that W^T W = C_p
            W = cholesky(C_p)
            detCp_log = chodet_from_tri(W, retlog=True)
            # solve for C_p d = L^T so that d = C_p^-1 L^T = C_q L^T
            d = chosolve_from_tri(W, self.larrTr)

        # multiply b d and a so that e = C^-1 L C_p^-1 L^T C^-1 y 
        e = np.dot(b, np.dot(d, a))
        # a minus e so that f = a - e = C^-1 y - C^-1 L C_p^-1 L^T C^-1 y
        #              thus f = C_v^-1 y
        f = a - e
        # multiply y^T  and f so that h = y^T C_v^-1 y
        h = np.dot(self.marr, f)
        # chi2_PRH = -0.5*h
        _chi2 = -0.5*h
        # following Carl Rasmussen's term, a penalty on the complexity of the model
        _compl_pen = -0.5*detC_log
        # penalty on blatant linear drift
        _wmean_pen = -0.5*detCp_log
        # final log_likelhood
        _log_like = _chi2 + _compl_pen + _wmean_pen
        if retq:
            q = np.dot(d, a)
            # q[0] to take the scalar value from the 0-d array
            return(_log_like, _chi2, _compl_pen, _wmean_pen, q[0])
        else:
            return(_log_like, _chi2, _compl_pen, _wmean_pen)




if __name__ == "__main__":    
#    zylclist= [
#               [[1.0, 2.0, 2.0], [5.0, 5.5, 4.8], [0.1, 0.1, 0.2]],
#               [[1.1, 2.1, 2.1], [5.1, 5.6, 4.9], [0.1, 0.1, 0.2]],
#              ]
    zylclist= [
               [[1.0, 2.0, 2.0], [5.0, 5.5, 4.8], [0.1, 0.1, 0.2]],
              ]
    zylc = zyLC(zylclist=zylclist)
    prh = PRH(zylc)
    

