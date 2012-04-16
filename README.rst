
=======
JAVELIN
=======


What is JAVELIN
===============

**JAVELIN** stands (reluctantly) for Just Another Version of Estimating Lags In
Nuclei. As an updated version of SPEAR, it is also completely re-written in
Python, providing much more flexibilities in both functionality and
visualization.

.. Caution::
    JAVELIN is still an ongoing project that has not reached a full release version yet.


Install JAVELIN
===============

Prerequisites
-------------

JAVELIN requires

#. `Fortran Compiler <http://en.wikipedia.org/wiki/Fortran>`_ (>F90)
#. `Python <http://python.org>`_ (>2.5)
#. `Numpy <http://numpy.org>`_ (>1.4)
#. `Scipy <http://scipy.org>`_ (>0.1)
#. `Matplotlib <http://matplotlib.sourceforge.net/>`_ (>1.0)

We strongly recommend that you have ``Lapack`` and ``Atlas`` library installed
in the system, although they are not necessary. It requires no extra effort to
install them as many systems either come with LAPACK and BLAS pre-installed
(MAC), or have them conveniently in software repositaries (Linux distributions).


Installation
------------

You can install JAVELIN by the standard Python package installation procedure::

    python setup.py config_fc --fcompiler=intel  install

or if you want to install the package to a specified directory ``JAVELINDIR``::

    python setup.py config_fc --fcompiler=intel install --prefix=JAVELINDIR

where ``config_fc --fcompiler=intel`` tells Python to use the *intel fortran
compiler* to compile Fortran source codes, you can also specifies other fortran
compilers that are available in your system, e.g.,::

    python setup.py config_fc --fcompiler=gnu95 install --prefix=JAVELINDIR

uses ``GFortran`` as its Fortran compiler.

Note that the short names for Fortran compilers may vary from system to system,
you can check the list of available Fortran compilers in your system using::

    python setup.py config_fc --help-fcompiler

and you can find them in the ``Fortran compilers found:`` section of the output.


Test Installation
-----------------

After installing JAVELIN, navigate into the ``examples`` directory::

    cd javelin/examples/

you can try::

    python demo.py test

to make sure the code works. Also, to test whether the graphics work, you can
try::

    python plotcov.py



.. image:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/covdemo.png
   :height: 400px
   :width:  400 px
   :alt: output of python plotcov.py
   :align: right



Demonstration
=============

In this section, we will quickly go through the underlying methodology of JAVELIN
using the example included in the ``examples/dat`` directory, where several
simulated light curves and MCMC chains are stored. We will describe and use each
file in turn, by running::

    python demo.py show

on the command line.

Variability Signals
-------------------

The first figure












