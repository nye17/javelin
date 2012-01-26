
=======
JAVELIN
=======


What is JAVELIN
===============

JAVELIN stands (reluctantly) as Just Another Version of Estimating LIne
reverberatioN. As a lighter version of SPEAR, it is also completely re-written
in Python, providing much more flexibility in extension and visualization.

JAVELIN is still an ongoing project that has not reached a full release version yet.


Install JAVELIN
===============

Prerequisites
-------------

JAVELIN requires

# Python (>2.5)
# Fortran Compiler (>F90)
# Numpy (>3.0)
# Scipy (>1.0)

and it is recommended that you have ``Lapack`` or even ``Atlas`` library
installed in the system, although they are not necessary.

Installation
------------

You can install JAVELIN by the standard Python package installation procedure::

    python setup.py config_fc --fcompiler=intel  install

or if you want to install the package to a specified directory::

    python setup.py config_fc --fcompiler=intel install --prefix="JAVLINDIR"

where ``config_fc --fcompiler=intel`` tells Python to use the *intel fortran
compiler* to compile Fortran source codes, you can also specifies other fortran
compilers that are available in your system, e.g.,::

    python setup.py config_fc --fcompiler=gnu95 install --prefix="JAVLINDIR"

uses ``GFortran`` as its Fortran compiler.

Note that the short names for Fortran compilers may vary from system to system,
you can check the list of available Fortran compilers in your system using::

    python setup.py config_fc --help-fcompiler

and you can find them in the ``Fortran compilers found:`` section of the output.


Test Installation
=================

After installing JAVLIN, you can try::

    testJAVELIN

to make sure everything works.









