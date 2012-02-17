#Last-modified: 17 Feb 2012 04:23:27 PM
import numpy as np
from predict import Predict
from psd import psd
from lcio import *
from zylc import zyLC, get_data
import matplotlib.pyplot as plt

"""
Test from scratch.
"""
def file_exists(fname) :
    try :
        f = open(fname, "r")
        f.close()
        return(True)
    except :
        return(False)


def main(set_plot=True):
    # create a `truth' mode light curve set with one continuum and two lines
    # object name: loopdeloop
    # line1 : yelm
    # line2 : zuunium
    sigma, tau = (0.10, 100.0)
    lagy, widy, scaley = (150.0, 3.0, 2.0)
    lagz, widz, scalez = (200.0, 6.0, 0.5)
    lags   = [0.0,   lagy,   lagz]
    wids   = [0.0,   widy,   widz]
    scales = [1.0, scaley, scalez]
    pass
    

if __name__ == "__main__":    
    main()
