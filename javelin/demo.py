#Last-modified: 12 Apr 2012 10:39:01 PM
import numpy as np
from predict import PredictSignal, PredictRmap, generateLine, generateError
from psd import psd
from lcio import *
from zylc import LightCurve, get_data
from lcmodel import Cont_Model, Rmap_Model
import matplotlib.pyplot as plt

"""
Test from scratch.
"""

names  = ["Continuum", "Yelm", "Zing"]
jdense = np.linspace(0.0, 2000.0, 4000)
sigma, tau = (2.00, 100.0)
tau_cut = 10.0
lagy, widy, scaley = (120.0,  3.0, 2.0)
lagz, widz, scalez = (250.0,  9.0, 0.5)
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
    else :
        raise RuntimeError("current no such covfunc implemented %s"%covfunc)
    # generate signal with no error
    edense = np.zeros_like(jdense)
    sdense = PS.generate(jdense, ewant=edense)
    imin = np.searchsorted(jdense, jmin)
    imax = np.searchsorted(jdense, jmax)
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

def getTrue(trufile, set_plot=False):
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
    return(zydata)

def getMock(zydata, confile, topfile, doufile, set_plot=False) :
    # downsample the truth to get more realistic light curves
    zydata_dou = True2Mock(zydata, lcmeans=lcmeans, sparse=[20, 20, 20], 
        errfac=[0.05, 0.05, 0.05], set_seasongap=True)
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

def fitCon(confile, confchain, names=None, threads=1, set_plot=False, nwalkers=100,
        nburn=50, nchain=50, figext=None) :
    zydata = get_data(confile, names=names)
    cont   = Cont_Model(zydata, "drw")
    if file_exists(confchain) :
        cont.load_chain(confchain)
    else :
        cont.do_mcmc(nwalkers=nwalkers, nburn=nburn, nchain=nchain, fburn=None,
                fchain=confchain, threads=threads)
    if set_plot :
        cont.show_hist(bins=100, figext=figext)
    return(cont.hpd)


def fitLag(linfile, linfchain, conthpd, names=None, 
        laglimit="baseline", lagrange=[100, 300], lagbinsize=1, 
        threads=1, set_plot=False, nwalkers=100, nburn=50, nchain=50,
        figext=None) :
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    if file_exists(linfchain) :
        rmap.load_chain(linfchain, set_verbose=False)
    else :
        rmap.do_mcmc(conthpd=conthpd, laglimit=laglimit, 
                nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=linfchain, threads=threads)
    if set_plot :
        rmap.break_chain([lagrange,]*(zydata.nlc-1))
        rmap.get_hpd()
        rmap.show_hist(bins=100, lagbinsize=lagbinsize, figext=figext)
    return(rmap.hpd)


if __name__ == "__main__":    
    set_plot = True
    threads  = 2
    # generate truth drw signal
    trufile = "dat/trulc.dat"
#    zydata  = getTrue(trufile, set_plot=set_plot)

    # generate mock light curves
    confile = "dat/loopdeloop_con.dat"
    topfile = "dat/loopdeloop_con_y.dat"
    doufile = "dat/loopdeloop_con_y_z.dat"
#    getMock(zydata, confile, topfile, doufile, set_plot=set_plot)

    # fit continuum
    confchain = "dat/chain0.dat"
    conthpd = fitCon(confile, confchain, threads=threads, set_plot=set_plot)

    # fit tophat
    topfchain = "dat/chain1.dat"
    tophpd = fitLag(topfile, topfchain, conthpd, threads=threads, set_plot=set_plot)

    # fit douhat
    doufchain = "dat/chain2.dat"
    douhpd = fitLag(doufile, doufchain, conthpd, threads=threads, set_plot=set_plot)




