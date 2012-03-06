import numpy as np
import javelin.lcio as IO
from javelin.zylc import LightCurve

def get_data(lcfile):
    lclist = IO.readlc_3c(lcfile)
    zydata = LightCurve(lclist)
    return(zydata)
