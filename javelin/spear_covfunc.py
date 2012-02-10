#Last-modified: 10 Feb 2012 02:13:38 AM


""" The SPEAR covariance function, called by gp.Covariance.
"""

# Requirements
# 1. called via spear_covfunc(x, y=None, **params)
# 2. has attribute 'diag_call': spear_covfunc(x, **params)
# 3. needs to resolve the covariance btw os_mesh and x
# 4. needs to take in 'symm'
