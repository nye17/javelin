#Last-modified: 04 Mar 2012 12:49:08 AM
import numpy as np
from predict import PredictSignal, PredictRmap, generateLine, generateError
from psd import psd
from lcio import *
from zylc import zyLC, get_data
import matplotlib.pyplot as plt

"""
Test from scratch.
"""

names  = ["cont", "Yelm", "Zing"]
jdense = np.linspace(0.0, 1000.0, 2000)
sigma, tau = (2.00, 100.0)
lagy, widy, scaley = (150.0,  3.0, 2.0)
lagz, widz, scalez = (200.0,  9.0, 0.5)
lags   = [0.0,   lagy,   lagz]
wids   = [0.0,   widy,   widz]
scales = [1.0, scaley, scalez]
lcmeans= [10.0,  15.0,    5.0]

def file_exists(fname) :
    try :
        f = open(fname, "r")
        f.close()
        return(True)
    except IOError:
        return(False)

def generateTrueLC():
    # create a `truth' mode light curve set with one continuum and two lines
    # object name: loopdeloop
    # line1 : yelm
    # line2 : zing
    jmin = np.max(lags)
    jmax = np.max(jdense)-1.0

    zylist = []
    # underlying continuum variability
    PS = PredictSignal(lcmean=0.0, covfunc="drw", sigma=sigma, tau=tau)
    # generate signal with no error
    edense = np.zeros_like(jdense)
    sdense = PS.generate(jdense, ewant=edense)
    imin = np.searchsorted(jdense, jmin)
    imax = np.searchsorted(jdense, jmax)
    print(imin),
    print(imax)
    zylist.append([jdense[imin: imax], sdense[imin: imax], edense[imin: imax]])

    for i in xrange(1, 3) :
        lag  = lags[i]
        wid  = wids[i]
        scale= scales[i]
        jl, sl = generateLine(jdense, sdense, lag, wid, scale,
                mc_mean=0.0, ml_mean=0.0)
        imin = np.searchsorted(jl, jmin)
        imax = np.searchsorted(jl, jmax)
        zylist.append([jl[imin: imax], sl[imin: imax], edense[imin: imax]])
    zydata = zyLC(zylist, names=names)
    return(zydata)

def True2Mock(zydata, lcmeans=lcmeans, sparse=[2, 4, 4], errfac=[0.05, 0.05,
    0.05], set_seasongap=True):
    zylclist= zydata.zylclist
    names   = zydata.names
    if set_seasongap :
        rj = zydata.rj
        j0 = zydata.jarr[0]
        j1 = zydata.jarr[-1]
        ng = np.floor(rj/180.0)
    zylclist_new = []
    for i in xrange(zydata.nlc):
        ispa = np.arange(0, zydata.nptlist[i], sparse[i])
        j = zydata.jlist[i][ispa]
        # strip off gaps
        if set_seasongap :
            dj = np.floor((j - j0)/180.0)
            igap = np.where(np.mod(dj, 2) == 0)
            indx = ispa[igap]
        else :
            indx = ispa
        j = zydata.jlist[i][indx]
        m = zydata.mlist[i][indx]
        e = zydata.elist[i][indx]
        # adding errors
        e = e*0.0+lcmeans[i]*errfac[i]
        et= generateError(e, errcov=0.0)
        m = m + et + lcmeans[i]
        zylclist_new.append([j,m,e])
    zymock = zyLC(zylclist_new, names=names)
    return(zymock)


if __name__ == "__main__":    
    set_plot = True
    # truth
    trufile = "dat/trulc.dat"
    if file_exists(trufile) :
        zydata = get_data(trufile, names=names)
    else :
        zydata = generateTrueLC()
        zydata.save(trufile)
    if set_plot :
        zydata.plot()

    # downsample the truth to get more realistic light curves
    confile = "dat/loopdeloop_con.dat"
    topfile = "dat/loopdeloop_con_y.dat"
    doufile = "dat/loopdeloop_con_y_z.dat"
    zydata_dou = True2Mock(zydata, lcmeans=lcmeans, sparse=[20, 20, 20], 
        errfac=[0.05, 0.05, 0.05], set_seasongap=False)
    zydata_dou.save_continuum(confile)
    zydata_dou.save(doufile)
    zylclist_top = zydata_dou.zylclist[:2]
    zydata_top = zyLC(zylclist_top)
    zydata_top.save(topfile)
    if set_plot :
        zydata_top.plot()
        zydata_dou.plot()

    plt.show()
