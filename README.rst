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
hosted by an imaginary `Loopdeloop galaxy <http://www.mariowiki.com/Loopdeeloop_Galaxy>`_, where two emission lines are
observed, `Ylem <http://en.wikipedia.org/wiki/Ylem>`_ and Zing.  Every file and
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
from the DRW model with time scale 100 days and variability amplitude sigma=2
mag.


.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/signal.png
   :scale: 80%

   Fig. 2: True light curves of loopdeeloop (from top to bottom: the Zing
   emission line, the Ylem emission line, and the continuum).

In practice, what we could observe are down-sampled versions of the true light
curves, sometimes with seasonal gaps because of the conflict with our Sun's
schedule, as shown by Fig. 3.

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mocklc.png
   :scale: 80%

   Fig. 3: Same as Fig. 2, but observed versions.

To directly derive lags from those sparse light curves is hard with traiditional
cross-correlation based methods. JAVELIN makes it much less formidable, by
incorporating the statistical properties of the continuum light curve into the
lag determination. Thus we need to run a continuum model to determine the DRW
paramters of the continuum light curve. Fig. 4 shows the posterior distribution
of the two DRW parameters of the continuum variability as calculated from
JAVELIN,

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc0.png
   :scale: 80%

   Fig. 4: Posterior distributions of the DRW parameters.

Once we derive the poseteriors of the DRW parameters, we then have a pretty good
idea of how much the continuum light curves in unobserved epochs should vary
relative to observed epochs, i.e., we know how to statistically interpolate the
continuum light curve. To measure the lag between the continuum and the Ylem
light curve, JAVELIN then tries to interpolate the continuum light curve based
on the posteriors derived in Fig. 4, and then shift, smooth, and scale each
interpolated continuum light curve to compare to the observed Ylem light curve.
After doing this try-and-err many many time in a MCMC run, JAVELIN finally
derives the posterior distribution of the lag t, the tophat width w, and the
scale factor s of the emission line, along with updated posteriors for the
timescale tau and the amplitude sigma of the continuum, as shown in Fig. 5.

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc1.png
   :scale: 100%

   Fig. 5: Posterior distributions of the emission line lag t, tophat width w,
   and the scale factor s for the Ylem light curve (bottom), with the top two
   panles showing the updated posteriors for tau and sigma.

However, we can see two peaks for the lag distribution in Fig. 5, which is
caused by the 180-day seaonal gaps in the two light curves - JAVELIN found that
it is much easier to shift the continuum by 180 days to compare to the line
light curve - there is no overlap between the two, therefore no objection from
the data!


Fortunately, we also have observations of the Zing light curve. Although equally
sparsely sampled with gaps inside, the mere existence of the Zing light curve
makes it impossible for JAVELIN to shift the contiuum by 180 days TWICE to
compare to the two line light curves! After another MCMC run, JAVELIN is able to
eliminate the second peak at 180 days and solve the lags for both emission lines
simultaneously, as shown in Fig. 6.

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/mcmc2.png
   :scale: 100%

   Fig. 6: Similar as Fig. 5, but after running JAVELIN with all three light
   curves simultaneously.

Finally, we want to know how the best--fit parameters from the last
MCMC run look like. It is generally very hard to visualize the fit for the
traditional cross-correlation methods, but JAVELIN is exceptionally good at
this - afterall all it has been doing is to interpolate and align light curves,
so why not for the best-fit parameters? Fig. 7 compares the best-fit light
curves and the observed ones shown earlier in Fig. 3. Apparently JAVELIN does a
great job of recovering the true light curves (compare to Fig. 2).

.. figure:: http://bitbucket.org/nye17/javelin/raw/default/examples/figs/prediction.png
   :scale: 80%

   Fig. 7: Comparion between the simulated light curves as computed from the
   best-fit parameters, and the observed light curves.












