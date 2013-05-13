#Last-modified: 03 May 2013 02:52:15 PM
import numpy as np
import matplotlib.pyplot as plt
from javelin.predict import PredictSignal, PredictRmap, generateLine, generateError
from javelin.lcio import *
from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model

"""
Test from scratch.
"""

#************** DO NOT EDIT THIS PART************
# names of the example light curves
names  = ["Continuum", "Yelm", "Zing"]
# smapling properties of the underlying signal
jdense = np.linspace(0.0, 2000.0, 4000)
# DRW parameters
sigma, tau = (2.00, 100.0)
# line parameters
lagy, widy, scaley = (120.0,  3.0, 2.0)
lagz, widz, scalez = (250.0,  9.0, 0.5)
lags   = [0.0,   lagy,   lagz]
wids   = [0.0,   widy,   widz]
scales = [1.0, scaley, scalez]
lcmeans= [10.0,  15.0,    5.0]
#************************************************

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

def getTrue(trufile, set_plot=False, mode="test"):
    if mode == "test" :
        return(None)
    elif mode == "show" :
        print("read true light curve signal from %s"%trufile)
        zydata = get_data(trufile, names=names)
    elif mode == "run" :
        print("generate true light curve signal")
        zydata = generateTrueLC(covfunc="drw")
        print("save true light curve signal to %s"%trufile)
        trufile = ".".join([trufile, "myrun"])
        zydata.save(trufile)
    if set_plot :
        print("plot true light curve signal")
        zydata.plot()
    return(zydata)

def getMock(zydata, confile, topfile, doufile, set_plot=False, mode="test") :
    # downsample the truth to get more realistic light curves
    if mode == "test" :
        return(None)
    else :
        zydata_dou = True2Mock(zydata, lcmeans=lcmeans, sparse=[20, 20, 20], 
            errfac=[0.05, 0.05, 0.05], set_seasongap=True)
        zylclist_top = zydata_dou.zylclist[:2]
        zydata_top = LightCurve(zylclist_top, names=names[0:2])
        if mode == "run" :
            confile = ".".join([confile, "myrun"])
            doufile = ".".join([doufile, "myrun"])
            topfile = ".".join([topfile, "myrun"])
            zydata_dou.save_continuum(confile)
            zydata_dou.save(doufile)
            zydata_top.save(topfile)
    if set_plot :
        print("plot mock light curves for continuum, yelm, and zing lines")
        zydata_dou.plot()

def fitCon(confile, confchain, names=None, threads=1, set_plot=False, nwalkers=100,
        nburn=50, nchain=50, figext=None, mode="test") :
    if mode == "run" :
        confile = ".".join([confile, "myrun"])
    zydata = get_data(confile, names=names)
    cont   = Cont_Model(zydata, "drw")
    if mode == "test" :
        print(cont([np.log(2.), np.log(100)], set_retq=True))
        return(None)
    elif mode == "show" :
        cont.load_chain(confchain)
    elif mode == "run" :
        confchain = ".".join([confchain, "myrun"])
        cont.do_mcmc(nwalkers=nwalkers, nburn=nburn, nchain=nchain, fburn=None,
                fchain=confchain, threads=1)
    if set_plot :
        cont.show_hist(bins=100, figext=figext)
    return(cont.hpd)

def fitLag(linfile, linfchain, conthpd, names=None, 
        lagrange=[50, 350], lagbinsize=1, 
        threads=1, set_plot=False, nwalkers=100, nburn=50, nchain=50,
        figext=None, mode="test") :
    if mode == "run" :
        linfile = ".".join([linfile, "myrun"])
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    if mode == "test" :
        if zydata.nlc == 2 :
            print(rmap([np.log(2.), np.log(100), lagy, widy, scaley ]))
        elif zydata.nlc == 3 :
            print(rmap([np.log(2.), np.log(100), lagy, widy, scaley, lagz, widz,
                scalez]))
        return(None)
    elif mode == "show" :
        rmap.load_chain(linfchain, set_verbose=False)
    elif mode == "run" :
        laglimit = [[0.0, 400.0],]*(zydata.nlc-1)
        print(laglimit)
#        laglimit = "baseline"
        linfchain = ".".join([linfchain, "myrun"])
        rmap.do_mcmc(conthpd=conthpd, laglimit=laglimit, 
                nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=linfchain, threads=threads)
    if set_plot :
        rmap.break_chain([lagrange,]*(zydata.nlc-1))
        rmap.get_hpd()
        rmap.show_hist(bins=100, lagbinsize=lagbinsize, figext=figext)
    return(rmap.hpd)

def showfit(linhpd, linfile, names=None, set_plot=False, mode="test") :
    if mode == "run" :
        linfile = ".".join([linfile, "myrun"])
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    if mode == "test" :
        return(None)
    else :
        zypred = rmap.do_pred(linhpd[1,:])
        zypred.names = names
    if set_plot :
        zypred.plot(set_pred=True, obs=zydata)

if __name__ == "__main__":    
    import sys
    try:
        import multiprocessing
        threads = multiprocessing.cpu_count()
    except (ImportError,NotImplementedError):
        threads = 1
        pass
    if threads > 1 :
        print("use multiprocessing on %d cpus"%threads)
    else :
        print("use single cpu")

    mode = sys.argv[1]
    if mode == "test" :
        set_plot = False
    elif mode == "show" :
        set_plot = True
    elif mode == "run" :
        set_plot = True

    # specified file names
    trufile   = "dat/trulc.dat"
    confile   = "dat/loopdeloop_con.dat"
    topfile   = "dat/loopdeloop_con_y.dat"
    doufile   = "dat/loopdeloop_con_y_z.dat"
    confchain = "dat/chain0.dat"
    topfchain = "dat/chain1.dat"
    doufchain = "dat/chain2.dat"

    # generate truth drw signal
    zydata  = getTrue(trufile, set_plot=set_plot, mode=mode)

    # generate mock light curves
    getMock(zydata, confile, topfile, doufile, set_plot=set_plot, mode=mode)

    # fit continuum
    conthpd = fitCon(confile, confchain, names=names[0:1],
            threads=threads, set_plot=set_plot, mode=mode)

    # fit tophat
    tophpd = fitLag(topfile, topfchain, conthpd, names=names[0:2],
            threads=threads, set_plot=set_plot, mode=mode)

    # fit douhat
    douhpd = fitLag(doufile, doufchain, conthpd, names=names,
        threads=threads, set_plot=set_plot, mode=mode)

    # show fit
    showfit(douhpd, doufile, names=names, set_plot=set_plot, mode=mode)




