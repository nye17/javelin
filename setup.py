#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
import distutils
import shutil
import os, sys


class MyClean(distutils.command.clean.clean):
    '''
    Subclass to remove any files created in an inplace build.

    This subclasses distutils' clean because neither setuptools nor
    numpy.distutils implements a clean command.

    '''
    def run(self):
        distutils.command.clean.clean.run(self)
        # Clean any build or dist directory
        if os.path.isdir("build"):
            shutil.rmtree("build", ignore_errors=True)
        if os.path.isdir("dist"):
            shutil.rmtree("dist", ignore_errors=True)


config = Configuration('javelin',parent_package=None,top_path=None)

# finding sdist or bdist
dist = "null"
for arg in sys.argv :
    if "dist" in arg :
        dist = arg



# ==============================
# = Compile Fortran extensions =
# ==============================


# I have disabled linking to the system lapack, quite a headache it seems for inexperienced users.
lapack_info = get_info('',1)

# For advanced users, if optimized lapack/ BLAS libraries are present, you can uncomment the following line to compile distributions that involve linear algebra against those.
# lapack_info = get_info('lapack_opt',1)




# ===========================================
# = Compile GP package's Fortran extensions =
# ===========================================

# Compile linear algebra utilities
if lapack_info:
    config.add_extension(name='gp.linalg_utils',sources=['javelin/gp/linalg_utils.f','javelin/blas_wrap.f'], extra_info=lapack_info)
    config.add_extension(name='gp.incomplete_chol',sources=['javelin/gp/incomplete_chol.f'], extra_info=lapack_info)

if not lapack_info or dist in ['bdist', 'sdist']:
    print('No optimized BLAS or Lapack libraries found, building from source. This may take a while...')
    f_sources = ['javelin/blas_wrap.f']
    for fname in os.listdir('blas/BLAS'):
        if fname[-2:]=='.f':
            f_sources.append('blas/BLAS/'+fname)

    for fname in ['dpotrs','dpotrf','dpotf2','ilaenv','dlamch','ilaver','ieeeck','iparmq']:
        f_sources.append('lapack/double/'+fname+'.f')

    config.add_extension(name='gp.linalg_utils',sources=['javelin/gp/linalg_utils.f'] + f_sources)
    config.add_extension(name='gp.incomplete_chol',sources=['javelin/gp/incomplete_chol.f'] + f_sources)


# Compile covariance functions
config.add_extension(name='gp.cov_funs.isotropic_cov_funs', sources=['javelin/gp/cov_funs/isotropic_cov_funs.f','blas/BLAS/dscal.f'], extra_info=lapack_info)

config.add_extension(name='gp.cov_funs.distances',sources=['javelin/gp/cov_funs/distances.f'], extra_info=lapack_info)

config.add_extension(name='spear_covfunc',sources=['javelin/spear_covfunc.f90'], extra_info=lapack_info)


config_dict = config.todict()
try:
    config_dict.pop('packages')
except:
    pass



if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(  version="0.35",
            description="JAVELIN: Python Version of SPEAR",
            author="Ying Zu",
            author_email="zuying@gmail.com ",
            url="https://bitbucket.org/nye17/javelin",
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
            cmdclass = {
                # Use our customized commands
                'clean': MyClean,
                },
            requires=['NumPy (>=1.3)', 'Python (>=3.0)',],
            long_description="""
            Ongoing effort to combine an integral GP module to the AGN variablility study.
            """,
            packages=["javelin",
                      "javelin/gp",
                      "javelin/gp/cov_funs",
                      "javelin/emcee_internal",
                      ],
            **(config_dict)
            )
