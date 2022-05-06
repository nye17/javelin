# 01 Sep 2021 18:10:57
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from javelin.predict import PredictSignal, PredictRmap, generateLine, generateError, PredictSpear
from javelin.lcio import *
from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model, DPmap_Model


def test_dpmap():
    tag = '.myrun'
    # tag = '.short'
    # tag = '.shorter'
    phofile   = "dat/loopdeloop_con_yb.dat" + tag
    # dphfile   = "dat/loopdeloop_con_yzb.dat" + tag
    zydata = get_data(phofile)
    # zydata = get_data(dphfile)
    # zydata.plot(marker="None", ms=1.0, ls="-", lw=2, figout="test", figext='pdf')
    dpmap   = DPmap_Model(zydata)
    # print(dpmap([np.log(3.), np.log(400), 100, 2, 0.5, 0, 0, 1]))
    # print(dpmap([np.log(3.), np.log(400), 100, 2, 0.5, 250, 0, 0.5]))
    # print(dpmap([np.log(3.), np.log(400), 250, 0, 0.5, 100, 2, 0.5]))
    print((dpmap([np.log(3.), np.log(400), 250, 2, 0.5, 100, 2, 0.5])))
    # print(dpmap([np.log(3.), np.log(400), 250, 0.011, 0.5, 100, 2, 0.5]))
    # print(dpmap([np.log(3.), np.log(400), 250.0, 4.0, 0.5, 100.0, 2, 0.5]))
    # pmap   = Pmap_Model(zydata)
    # print(pmap([np.log(3.), np.log(400), 100, 2, 0.5, 1]))


if __name__ == "__main__":
    test_dpmap()
