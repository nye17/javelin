#!/bin/sh

# on my Debian laptop
python setup.py install --prefix=~/usr sdist
#python setup.py config_fc --fcompiler=intelem install --prefix="~/usr" sdist

# on arjuna
#python setup.py config_fc --fcompiler=intelem install --prefix=~/local