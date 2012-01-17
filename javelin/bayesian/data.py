import numpy as np
import javelin.lcio as IO
from javelin.zylc import zyLC

def get_data(lcfile):
    lclist = IO.readlc_3c(lcfile)
    zydata = zyLC(lclist)
    return(zydata)
