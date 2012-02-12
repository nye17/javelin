#Last-modified: 11 Feb 2012 11:14:32 PM


""" The SPEAR covariance function, called by gp.Covariance.
"""

# Requirements
# 1. called via spear_covfunc(x, y=None, **params)
# 2. has attribute 'diag_call': spear_covfunc(x, **params)
# 3. needs to resolve the covariance btw os_mesh and x
# 4. needs to take in 'symm'

# basic function is to take x, y (vectors or scalars), along with their ids,
# DRW parameters, transfer function parameters, etc., and return the matrix.

def spear(x, y=None, idx=None, idy=None, amp=1.0, scale=1.0, symm=None):
    pass
