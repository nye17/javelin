#!/bin/bash

name=`hostname`

python setup.py clean

if [ $name = "nanhu" ];then
    echo "nanhu"
    python setup.py install --user
else
    echo "Others"
    python setup.py install --user
fi

