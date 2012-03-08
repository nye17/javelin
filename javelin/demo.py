#Last-modified: 08 Mar 2012 02:00:32 AM
import numpy as np
from predict import PredictSignal, PredictRmap, generateLine, generateError
from psd import psd
from lcio import *
from zylc import LightCurve, get_data
import matplotlib.pyplot as plt

"""
Test from scratch.
"""

names  = ["Continuum", "Yelm", "Zing"]
jdense = np.linspace(0.0, 1000.0, 2000)
sigma, tau = (2.00, 100.0)
tau_cut = 10.0
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

def generateTrueLC(covfunc="drw"):
    # create a `truth' mode light curve set with one continuum and two lines
    # object name: loopdeloop
    # line1 : yelm
    # line2 : zing
    jmin = np.max(lags)
    jmax = np.max(jdense)-1.0

    zylist = []
    # underlying continuum variability
    if covfunc == "drw" :
        PS = PredictSignal(lcmean=0.0, covfunc=covfunc, sigma=sigma, tau=tau)
    elif covfunc == "kepler2_exp" :
        PS = PredictSignal(lcmean=0.0, covfunc=covfunc, sigma=sigma, tau=tau, 
                nu=tau_cut, rank="NearlyFull")
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
    zydata = LightCurve(zylist, names=names)
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
    zymock = LightCurve(zylclist_new, names=names)
    return(zymock)


if __name__ == "__main__":    
    set_plot = True
    # truth drw
    trufile = "dat/trulc.dat"
    if False :
        if file_exists(trufile) :
            print("read true light curve signal from %s"%trufile)
            zydata = get_data(trufile, names=names)
        else :
            print("generate true light curve signal")
            zydata = generateTrueLC(covfunc="drw")
            print("save true light curve signal to %s"%trufile)
            zydata.save(trufile)
        if set_plot :
            print("plot true light curve signal")
            zydata.plot()

    # truth kepler
    trufile_k2e = "dat/trulc_k2e.dat"
    if True :
        if file_exists(trufile_k2e) :
            print("read true k2e light curve signal from %s"%trufile_k2e)
            zydata_k2e = get_data(trufile_k2e, names=names)
        else :
            print("generate k2e true light curve signal")
            zydata_k2e = generateTrueLC(covfunc="kepler2_exp")
            print("save true k2e light curve signal to %s"%trufile)
            zydata_k2e.save(trufile_k2e)
        if set_plot :
            print("plot true k2e light curve signal")
            zydata_k2e.plot()


    confile = "dat/loopdeloop_con.dat"
    topfile = "dat/loopdeloop_con_y.dat"
    doufile = "dat/loopdeloop_con_y_z.dat"
    if False :
        # downsample the truth to get more realistic light curves
        zydata_dou = True2Mock(zydata, lcmeans=lcmeans, sparse=[20, 20, 20], 
            errfac=[0.05, 0.05, 0.05], set_seasongap=False)
        zydata_dou.save_continuum(confile)
        zydata_dou.save(doufile)
        zylclist_top = zydata_dou.zylclist[:2]
        zydata_top = LightCurve(zylclist_top, names=names[0:2])
        zydata_top.save(topfile)
        if set_plot :
            print("plot mock light curves for continuum and yelm line")
            zydata_top.plot()
            print("plot mock light curves for continuum, yelm, and zing lines")
            zydata_dou.plot()

    confile_k2e = "dat/loopdeeswoop_con.dat"
    topfile_k2e = "dat/loopdeeswoop_con_y.dat"
    doufile_k2e = "dat/loopdeeswoop_con_y_z.dat"
    if True :
        # downsample the truth to get more realistic light curves
        zydata_dou_k2e = True2Mock(zydata_k2e, lcmeans=lcmeans, sparse=[20, 20, 20], 
            errfac=[0.05, 0.05, 0.05], set_seasongap=False)
        zydata_dou_k2e.save_continuum(confile_k2e)
        zydata_dou_k2e.save(doufile_k2e)
        zylclist_top_k2e = zydata_dou_k2e.zylclist[:2]
        zydata_top_k2e = LightCurve(zylclist_top_k2e, names=names[0:2])
        zydata_top_k2e.save(topfile_k2e)
        if set_plot :
            print("plot k2e mock light curves for continuum and yelm line")
            zydata_top_k2e.plot()
            print("plot k2e mock light curves for continuum, yelm, and zing lines")
            zydata_dou_k2e.plot()

