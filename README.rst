.. role:: raw-math(raw)
    :format: latex html


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

    $ python setup.py config_fc --fcompiler=intel  install

or if you want to install the package to a specified directory ``JAVELINDIR``::

    $ python setup.py config_fc --fcompiler=intel install --prefix=JAVELINDIR

where ``config_fc --fcompiler=intel`` tells Python to use the *intel fortran
compiler* to compile Fortran source codes, you can also specifies other fortran
compilers that are available in your system, e.g.,::

    $ python setup.py config_fc --fcompiler=gnu95 install --prefix=JAVELINDIR

uses ``GFortran`` as its Fortran compiler.

Note that the short names for Fortran compilers may vary from system to system,
you can check the list of available Fortran compilers in your system using::

    $ python setup.py config_fc --help-fcompiler

and you can find them in the ``Fortran compilers found:`` section of the output.


Test Installation
-----------------

After installing JAVELIN, navigate into the ``examples`` directory::

    $ cd javelin/examples/

you can try::

    $ python demo.py test

to make sure the code works. Also, to test whether the graphics work, you can
try::

    $ python plotcov.py



.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/covdemo.png
   :scale: 80%

   Fig. 1 : Illustration of four continuum models available in JAVELIN.

which is exactly the Figure 1 in `Zu et al. (2012) <http://arxiv.org/abs/1202.3783>`_.



Demonstration
=============

Here we briefly explain how to use JAVELIN to caculate the line lags for the AGN
hosted by an imaginary `Loopdeloop galaxy
<http://www.mariowiki.com/Loopdeeloop_Galaxy>`_, where two emission lines are
observed, `Ylem <http://en.wikipedia.org/wiki/Ylem>` and Zing.  Every file and
script referred to here can be found inside ``examples`` directory::

    $ cd javelin/examples

To give you an idea of how things work in JAVELIN, let us first go through
several figures illustrating the underlying methodlogy of JAVELIN. You can also
show the figures below locally by running::

    $ python demo.py show

on the command line. 

We assume the quasar variability on scales longer than a few days can be well
described by a Damped random walk (DRW) model, and the emission line light
curves are simply the lagged, smoothed, and scaled versions of the continuum
light curve. Fig. 1 shows the true light curves for the continuum, the Ylem, and
the Zing lines. In particular, the Ylem (Zing) light curve is lagged by 120
(250) days, scaled by a factor of 3 (9), and smoothed by a top hat of width 3
(9) days, from the continuum light curve. The continuum light curve is generated
from the DRW model with time scale 100 days and variability amplitude
:math:`\sigma=2\,mag`.


.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/signal.png
   :scale: 80%

   Fig. 2: True light curves of loopdeeloop (from top to bottom: the Zing
   emission line, the Ylem emission line, and the continuum).

In practice, what we could observe are sparsely sampled versions of the true light
curves, sometimes with seasonal gaps because of the conflict with our Sun's
schedule, as shown by Fig. 3.

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mocklc.png
   :scale: 80%

   Fig. 3: Same as Fig. 2, but observed versions, with light curves.

.. image:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc0.png
   :scale: 80%

.. image:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc1.png
   :scale: 80%

.. image:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc2.png
   :scale: 80%

.. image:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/prediction.png
   :scale: 80%












