#!/bin/bash

name=`hostname`

python setup.py clean

if [ $name = "mitchell" ];then
    echo "mitchell"
    python setup.py config_fc --fcompiler=intel install --prefix="~/local" sdist
#    python setup.py config_fc --fcompiler=gnu95 install --prefix="~/local" sdist
elif [ $name = "Sing-Sing" ];then
    echo "sing-sing"
    python setup.py install --prefix="~/usr" sdist
elif [ $name = "arjuna.mps.ohio-state.edu" ];then
    echo "arjuna"
    echo "JAVELIN does not work in logon macihne, switch to a node"
elif [ `echo ${name} | grep -c "node"` -eq 1 ]; then
    echo "arjuna node"
    python setup.py config_fc --fcompiler=intelem install --prefix="~/local" sdist
fi 

