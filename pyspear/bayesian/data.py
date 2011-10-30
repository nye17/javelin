import pyspear.lcio as IO
from pyspear.zylc import zyLC
import numpy as np

#lcfile = "dat/mock_l100c1_t10s2n0.5.dat"

def get_data(lcfile):
    lclist = IO.readlc_3c(lcfile)
    zydata = zyLC(lclist)
    return(zydata)
