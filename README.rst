
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
observed, `Ylem <http://en.wikipedia.org/wiki/Ylem>`_ and Zing. If you are
already familiar with the `Zu et al. (2011) <http://arxiv.org/abs/1008.0641>`_
paper, feel free to skip to the next section.  Every file and script referred to
here can be found inside ``examples`` directory::

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
   :scale: 150%

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
   :scale: 150%

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



Usage
=====

To use JAVELIN, it is useful to have some a pirori knowledge of Python, but not
necessary. Here we will walk you through the actual procedures outlined in the
last section. In this section, we will manipulate the files in two different
terminals, one is the usual Unix command line marked by ``\$`` in the beginning,
one is the Python terminal started with ``>>>``. 

Reading Light Curves
--------------------

Starting from the data files in the ``examples/dat`` directly::

    $ cd javelin/examples/dat

JAVELIN could work on two types of light curve files, the first one is the typical
3-column file like ``con.dat``, ``yelm.dat``, and ``zing.dat`` in the current
directory. If you do::

    $ head -n 3 con.dat

to show the first 3 rows of the continuum light curve file ``con.dat``::
    
    250.06252   10.93763    0.50000
    260.06502   10.33037    0.50000
    270.06752   10.70079    0.50000

where the 1st, 2nd, and 3rd columns are *observing epoch*, *light curve value*,
and *measurement uncertainty*, respectively. Since the basic data unit in JAVELIN  is
a ``LightCurve`` object, you need to read the data files through a function
into the ``LightCurve`` object. Open a Python terminal in the ``dat`` directory
and do::

    >>>import javelin

so that you could call JAVELIN within current session of the Python terminal,
and then do::

    >>>from javelin.zylc import get_data
    >>>javdata = get_data(["con.dat", "yelm.dat"], names=["Continuum", "Yelm"])

to load the continuum light curve ``con.dat`` and the Yelm light curve
``yelm.dat`` into a ``LightCurve`` object called ``javdata``, with ``names`` as
"Continuum" and "Yelm". The brackets ``[]`` tell JAVELIN that the two light
curves should be analyzed in one set, and if you want to check out the light
curves in figures just run::

    >>>javdata.plot()

Note that in Python you have to keep the parentheses even no arguments are needed.


The second type of file JAVELIN likes is
a slight variant of the 3-column format, like ``loopdeloop_con.dat``,
``loopdeloop_con_y.dat``, and ``loopdeloop_con_y_z.dat`` in the current
directory. As suggested by the names of these files, since JAVELIN usually works
on several light curves simultaneously, it is useful (at least to me) to keep
different set of data files separated (similar to the brackets used in the
reading of 3-column files). 

Imagine you want to fit two light curves, the first one should always
be the continuum light curves and the second one be the line light curve. If the
continuum light curve has 5 data points while the line light curve has 4, the
data file should be like (texts after # are comments, not part of the file) ::

    2                       # number of light curves, continuum first
    5                       # number of data points in the first light curve
    461.5  22.48    0.36   # each light curve entry consists of "epoch", "value", and  "uncertainty"
    490.6  20.30    0.30
    520.3  19.59    0.56
    545.8  20.11    0.15
    769.6  21.12    1.20
    4                       # number of data points in the second light curve
    545.8   9.82    0.23
    890.4  11.86    0.58
    949.4  10.55    0.87
    988.6  11.06    0.27    

To read the second type of files, simply do::

    >>>javdata2 = get_data("loopdeloop_con_y.dat", names=["Continuum", "Yelm"])

Note right now there are only brakets from the ``names``, but a single string
for the input file. Given ``loopdeloop_con_y.dat`` is just another version of
packing ``con.dat`` and ``yelm.dat`` together, ``javdata`` and ``javedata2``
are equivalent to each other. You can varify this by doing ``javdata2.plot()``.


Fiting the Continuum
--------------------

As shown in the last section, we need to fit the continuum frist, i.e., work
with the continuum light curve alone to derive the posterior distributions of
DRW parameters. Since for now we only work on the continuum model, we can load
the continuum light curve either by::

    >>>javdata3 = get_data(["con.dat",], names=["Continuum",]) 

or by::

    >>>javdata3 = get_data("loopdeloop_con.dat", names=["Continuum",]) 

Note the brakets are still needed even for loading a single light curve.

After loading the data, we need to set up a continuum model. In JAVELIN the
light curve models are described in the ``javelin.lcmodel`` module, for now we
need to initiate the ``Cont_Model`` class::

    >>>from javelin.lcmodel import Cont_Model
    >>>cont = Cont_Model(javdata3)

Without exploring any further options, you can simply run::

    >>>>cont.do_mcmc(fchain="mychain0.dat")

to start a MCMC analysis and the chain will be saved into "mychain0.dat" file.
By default, the chain will go through 5000 iterations for burn-in period, and
then another 5000 iterations for the actual chain. JAVELIN uses the `kick-ass
MCMC sampler named emcee <http://danfm.ca/emcee/>`_ introduced by  `Dan
Foreman-Mackey et al (2012) <http://arxiv.org/abs/1202.3665>`. ``emcee`` works
by releasing numerous ``walkers`` at every possible corner of the parameter
space, which then collaboratively sample the posterior probability
distributions. The number of ``walkers``, the number of burn-in iterations, and
the number of sampling iterations for each ``walker`` are specified by ``nwalker``
(default: 100), ``nchain`` (default: 50), and ``nburn`` (default: 50),
respectively. For examples, if you want to double the chain length of both
burn-in and sampling periods (well, you do not want to do it right now)::

    >>>>cont.do_mcmc(nwalkers=100, nburn=100, nchain=100, fchain="mychain0_long.dat")

After sampling, you can check the 1D posterior distributions of tau and sigma::

    >>>cont.show_hist(bins=100)

which looks like Fig. 4.

The output ``fchain`` is simply a two-column txt file with the first column
log(sigma) and the second one log(tau), both natural logs.

Olders chains can be reloaded for analysis by::

    >>>cont.load_chain("mychain0.dat")

and the highest posterior density (HPD) intervals can be retrieved by::

    >>>conthpd = cont.hpd
    >>>print(conthpd)
    [[ 0.363  3.923]
     [ 0.518  4.29 ]
     [ 0.737  4.743]]

which is a 3x2 array with the three elements of the first(second) column being
the 18%, 50%, and 84% values for log sigma (log tau).



