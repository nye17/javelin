#!/usr/bin/env python
# Do not add setuptools here; use setupegg.py instead. Nose still has problems running
# tests inside of egg packages, so it is useful to be able to install without eggs as needed.
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
import os, sys

config = Configuration('pyspear',parent_package=None,top_path=None)
dist = sys.argv[1]


# ==============================
# = Compile Fortran extensions =
# ==============================

# If optimized lapack/ BLAS libraries are present, compile distributions that involve linear algebra against those.
# Otherwise compile blas and lapack from netlib sources.
lapack_info = get_info('lapack_opt',1)




# ===========================================
# = Compile GP package's Fortran extensions =
# ===========================================

# Compile linear algebra utilities
if lapack_info:
    config.add_extension(name='gp.linalg_utils',sources=['pyspear/gp/linalg_utils.f','pyspear/blas_wrap.f'], extra_info=lapack_info)
    config.add_extension(name='gp.incomplete_chol',sources=['pyspear/gp/incomplete_chol.f'], extra_info=lapack_info)

if not lapack_info or dist in ['bdist', 'sdist']:
    print 'No optimized BLAS or Lapack libraries found, building from source. This may take a while...'
    f_sources = ['pyspear/blas_wrap.f']
    for fname in os.listdir('blas/BLAS'):
        if fname[-2:]=='.f':
            f_sources.append('blas/BLAS/'+fname)

    for fname in ['dpotrs','dpotrf','dpotf2','ilaenv','dlamch','ilaver','ieeeck','iparmq']:
        f_sources.append('lapack/double/'+fname+'.f')

    config.add_extension(name='gp.linalg_utils',sources=['pyspear/gp/linalg_utils.f'] + f_sources)
    config.add_extension(name='gp.incomplete_chol',sources=['pyspear/gp/incomplete_chol.f'] + f_sources)


# Compile covariance functions
config.add_extension(name='gp.cov_funs.isotropic_cov_funs',\
sources=['pyspear/gp/cov_funs/isotropic_cov_funs.f','blas/BLAS/dscal.f'],\
extra_info=lapack_info)

config.add_extension(name='gp.cov_funs.distances',sources=['pyspear/gp/cov_funs/distances.f'], extra_info=lapack_info)


config_dict = config.todict()
try:
    config_dict.pop('packages')
except:
    pass



if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(  version="0.1alpha",
            description="Python Version of SPEAR",
            author="Ying Zu",
            author_email="zuying@gmail.com ",
            url="https://bitbucket.org/nye17/pyspear",
            license="Academic Free License",
            classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: Academic Free License (AFL)',
                'Programming Language :: Python',
                'Programming Language :: Fortran',
                'Topic :: Scientific/Engineering',
                 ],
            requires=['NumPy (>=1.3)',],
            long_description="""
            Ongoing effort to combine an integral GP module to the AGN variablility study.
            """,
            packages=["pyspear", 
#                      "pyspear/likelihood", 
#                      "pyspear/bayesian", 
                      "pyspear/examples/gp", 
                      "pyspear/gp", 
                      "pyspear/gp/cov_funs" ],
            **(config_dict)
            )

