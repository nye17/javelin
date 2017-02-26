#Last-modified: 08 Dec 2013 15:54:47
import numpy as np
import matplotlib.pyplot as plt
from javelin.predict import PredictSignal, PredictRmap, generateLine, generateError, PredictSpear
from javelin.lcio import *
from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model

""" Tests from scratch.
"""

#************** PLEASE DO NOT EDIT THIS PART*****
# show figures interactively
figext = None
# names of the true light curves
names  = ["Continuum", "Yelm", "Zing", "YelmBand"]
# dense sampling of the underlying signal
jdense = np.linspace(0.0, 2000.0, 2000)
# DRW parameters
sigma, tau = (3.00, 400.0)
# tau_cut
tau_cut = 7.0
# line parameters
lagy, widy, scaley = (100.0,   2.0, 0.5)
lagz, widz, scalez = (250.0,   4.0, 0.25)
lags   = [0.0,   lagy,   lagz]
wids   = [0.0,   widy,   widz]
scales = [1.0, scaley, scalez]
llags   = [   lagy,   lagz]
lwids   = [   widy,   widz]
lscales = [ scaley, scalez]
lcmeans= [10.0,  5.0,  2.5]
#************************************************

def file_exists(fname) :
    try :
        f = open(fname, "r")
        f.close()
        return(True)
    except IOError:
        return(False)

def getTrue(trufile, set_plot=False, mode="test", covfunc="drw"):
    """ Generating dense, error-free light curves as the input signal.

    Parameters
    ---
    trufile: str
        filename for reading/storing the light curve file.
    set_plot: bool
        Draw the light curves if True.
    mode: str
        mode of this example script: "test" is for doing nothing but to show the
        modules are correctly loaded; "run" is to run a full test by
        regenerating all the data files and results of this demo; and "show" is
        to load all the precalculated results and plot them.
    covfunc: str
        the covariance function for the underlying variability model. For this
        demo we have either the default "drw" model or the "kepler2_exp" model
        which mimics the short time scale cutoff seen in Kepler.

    """
    if mode == "test" :
        return(None)
    elif mode == "show" :
        print("read true light curve signal from %s"%trufile)
        zydata = get_data(trufile, names=names)
    elif mode == "run" :
        print("generate true light curve signal")
        if covfunc == "drw":
            print("generating DRW light curves...")
            # this is the fast way
            zydata = generateTrueLC2(covfunc="drw")
        else:
            print("generating Kepler light curves...")
            # slower
            zydata = generateTrueLC(covfunc="kepler2_exp")
        print("save true light curve signal to %s"%trufile)
        trufile = ".".join([trufile, "myrun"])
        zydata.save(trufile)
    if set_plot :
        print("plot true light curve signal")
        zydata.plot(marker="None", ms=1.0, ls="-", lw=2, figout="signal", figext=figext)
    return(zydata)

def generateTrueLC(covfunc="kepler2_exp"):
    """generateTrueLC

    covfunc : str, optional
        Name of the covariance funtion, "drw" or "kepler2_exp" (default: kepler2_exp).

    """
    # create a `truth' mode light curve set with one continuum and two lines
    # object name: loopdeloop
    # line1 : yelm
    # line2 : zing
    jmin = np.max(lags)
    jmax = np.max(jdense)-1.0
    zylist = []
    # this is for handling the prediction for Continuum.
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
    zylist.append([jdense[imin: imax], sdense[imin: imax]+lcmeans[0], edense[imin: imax]])
    # this is for handling the prediction for Yelm, and Zing.
    for i in xrange(1, 3) :
        lag  = lags[i]
        wid  = wids[i]
        scale= scales[i]
        jl, sl = generateLine(jdense, sdense, lag, wid, scale, mc_mean=0.0, ml_mean=0.0)
        imin = np.searchsorted(jl, jmin)
        imax = np.searchsorted(jl, jmax)
        # print np.mean(sl)
        zylist.append([jl[imin: imax], sl[imin: imax]+lcmeans[i], edense[imin: imax]])
        if i == 1:
            # special continuum prediction for YelmBand at the observed epochs of the YelmLine
            jdense_yb = jl[imin: imax]
            edense_yb = np.zeros_like(jdense_yb)
            sdense_yb = PS.generate(jdense_yb, ewant=edense_yb)
            phlc = [jdense_yb, sdense_yb + lcmeans[i] + zylist[1][1], edense_yb]
    # this is for handling the prediction for YelmBand.
    # combine into a single LightCurve
    zylist.append(phlc)
    zydata = LightCurve(zylist, names=names)
    return(zydata)

def generateTrueLC2(covfunc="drw"):
    """ Generate RMap light curves first, with the sampling designed to allow a post-processing into the line band light curve. The simlest solution is to build light curves on dense regular time axis. The only downside here is that, only 'drw' covariance is allowed.

    covfunc : str, optional
        Name of the covariance funtion (default: drw).

    """
    if covfunc != "drw" :
        raise RuntimeError("current no such covfunc implemented for generateTrueLC2, see demo_covfunc.py for details on how to generate LCs using non-DRW models %s"%covfunc)
    ps = PredictSpear(sigma, tau, llags, lwids, lscales, spearmode="Rmap")
    mdense = np.zeros_like(jdense)
    edense = np.zeros_like(jdense)
    zylistold = [[jdense, mdense+lcmeans[0], edense], [jdense, mdense+lcmeans[1], edense], [jdense, mdense+lcmeans[2], edense],]
    # this is for handling the prediction for Continuum, Yelm, and Zing.
    zylistnew = ps.generate(zylistold)
    # this is for handling the prediction for YelmBand.
    phlc = [jdense, mdense, edense]
    phlc[1] = zylistnew[0][1] + zylistnew[1][1]
    # combine into a single LightCurve
    zylistnew.append(phlc)
    zydata = LightCurve(zylistnew, names=names)
    return(zydata)

def True2Mock(zydata, sparse=[2, 4, 4], errfac=[0.01, 0.01, 0.01], hasgap=[True, True, True], errcov=0.0):
    """ Postprocess true light curves to observed light curves.

    Parameters
    ----------
    zydata: LightCurve
        Input true LightCurve.
    """
    test = np.array([len(sparse), len(errfac), len(hasgap)])  == zydata.nlc
    if not np.all(test) :
        raise RuntimeError("input dimensions do not match")
    zylclist= zydata.zylclist
    names   = zydata.names
    zylclist_new = []
    if any(hasgap) :
        # some may have gaps, to make sure the sun is synchronized in all light curves, we have to do this globally.
        rj = zydata.rj
        j0 = zydata.jarr[0]
        j1 = zydata.jarr[-1]
        ng = np.floor(rj/180.0)
    for i in xrange(zydata.nlc):
        ispa = np.arange(0, zydata.nptlist[i], sparse[i])
        j = zydata.jlist[i][ispa]
        # strip off gaps
        if hasgap[i] :
            dj = np.floor((j - j0)/180.0)
            igap = np.where(np.mod(dj, 2) == 0)
            indx = ispa[igap]
        else :
            indx = ispa
        j = zydata.jlist[i][indx]
        m = zydata.mlist[i][indx] + zydata.blist[i]
        e = zydata.elist[i][indx]
        # adding errors
        e = e*0.0+m*errfac[i]
        et= generateError(e, errcov=errcov)
        m = m + et
        zylclist_new.append([j,m,e])
    zymock = LightCurve(zylclist_new, names=names)
    return(zymock)

def getMock(zydata, confile, topfile, doufile, phofile, set_plot=False, mode="test") :
    """ downsample the truth to get more realistic light curves
    """
    if mode == "test" :
        return(None)
    else :
        _c, _y, _z, _yb = zydata.split()
        _zydata = _c + _y + _z
        zydata_dou = True2Mock(_zydata, sparse=[8, 8, 8], errfac=[0.01, 0.01, 0.01], hasgap=[True, True, True], errcov=0.0)
        zylclist_top = zydata_dou.zylclist[:2]
        zydata_top = LightCurve(zylclist_top, names=names[0:2])
        _zydata = _c + _yb
        zydata_pho = True2Mock(_zydata, sparse=[8, 8], errfac=[0.01, 0.01], hasgap=[True, True], errcov=0.0)
        if mode == "run" :
            confile = ".".join([confile, "myrun"])
            doufile = ".".join([doufile, "myrun"])
            topfile = ".".join([topfile, "myrun"])
            phofile = ".".join([phofile, "myrun"])
            zydata_dou.save_continuum(confile)
            zydata_dou.save(doufile)
            zydata_top.save(topfile)
            zydata_pho.save(phofile)
    if set_plot :
        print("plot mock light curves for continuum, yelm, zing, and yelm band lines")
        _c, _yb = zydata_pho.split()
        zymock = zydata_dou + _yb
        zymock.plot(figout="mocklc", figext=figext)

def fitCon(confile, confchain, names=None, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100, mode="test") :
    """ fit the continuum model.
    """
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
        cont.do_mcmc(nwalkers=nwalkers, nburn=nburn, nchain=nchain, fburn=None, fchain=confchain, threads=1)
    if set_plot :
        cont.show_hist(bins=100, figout="mcmc0", figext=figext)
    return(cont.hpd)

def fitLag(linfile, linfchain, conthpd, names=None, lagrange=[50, 300], lagbinsize=1, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100, mode="test") :
    """ fit the Rmap model.
    """
    if mode == "run" :
        linfile = ".".join([linfile, "myrun"])
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    if mode == "test" :
        if zydata.nlc == 2 :
            print(rmap([np.log(2.), np.log(100), lagy, widy, scaley ]))
        elif zydata.nlc == 3 :
            print(rmap([np.log(2.), np.log(100), lagy, widy, scaley, lagz, widz, scalez]))
        return(None)
    elif mode == "show" :
        rmap.load_chain(linfchain, set_verbose=False)
    elif mode == "run" :
        laglimit = [[0.0, 400.0],]*(zydata.nlc-1)
        print(laglimit)
#        laglimit = "baseline"
        linfchain = ".".join([linfchain, "myrun"])
        rmap.do_mcmc(conthpd=conthpd, lagtobaseline=0.5, laglimit=laglimit,
                nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=linfchain, threads=threads)
    if set_plot :
        rmap.break_chain([lagrange,]*(zydata.nlc-1))
        rmap.get_hpd()
        if zydata.nlc == 2 :
            figout = "mcmc1"
        else :
            figout = "mcmc2"
        rmap.show_hist(bins=100, lagbinsize=lagbinsize, figout=figout, figext=figext)
    return(rmap.hpd)

def fitPmap(phofile, phofchain, conthpd, names=None, lagrange=[50, 300], lagbinsize=1, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100,mode="test") :
    """ fit the Pmap model.
    """
    if mode == "run" :
        phofile = ".".join([phofile, "myrun"])
    zydata = get_data(phofile, names=names)
    pmap   = Pmap_Model(zydata)
    if mode == "test" :
        print(pmap([np.log(2.), np.log(100), lagy, widy, scaley,  1.0]))
        return(None)
    elif mode == "show" :
        pmap.load_chain(phofchain, set_verbose=False)
    elif mode == "run" :
        laglimit = [[50.0, 130.0]] # XXX here we want to avoid 180 day limit.
        widlimit = [[0, 7.0]] # XXX here we want to avoid long smoothing width
        phofchain = ".".join([phofchain, "myrun"])
        pmap.do_mcmc(conthpd=conthpd, lagtobaseline=0.5, laglimit=laglimit,
                widlimit=widlimit, nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=phofchain, threads=threads)
    if set_plot :
        pmap.break_chain([lagrange,])
        pmap.get_hpd()
        pmap.show_hist(bins=100, lagbinsize=lagbinsize, figout="mcmc3", figext=figext)
    return(pmap.hpd)

def showfit(linhpd, linfile, names=None, set_plot=False, mode="test") :
    if mode == "run" :
        linfile = ".".join([linfile, "myrun"])
    print linfile
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    if mode == "test" :
        return(None)
    else :
        zypred = rmap.do_pred(linhpd[1,:])
        zypred.names = names
    if set_plot :
        zypred.plot(set_pred=True, obs=zydata, figout="prediction", figext=figext)

def demo(mode, covfunc="drw") :
    """ Demonstrate the main functionalities of JAVELIN.

    Parameters
    ----------

    mode: string
        "test" : just go through some likelihood calculation to make sure JAVELIN is correctly installed.
        "show" : load example light curves and chains and show plots.
        "run" : regenerate all the light curves and chains.

    """
    from sys import platform as _platform
    if True :
        if mode   == "test" :
            set_plot = False
        elif mode == "show" :
            set_plot = True
        elif mode == "run" :
            set_plot = True
        try :
            import multiprocessing
            if _platform == "darwin" :
                # for some reason, Mac cannot handle the pools in emcee.
                threads = 1
            else :
                threads = multiprocessing.cpu_count()
        except (ImportError,NotImplementedError) :
            threads = 1
        if threads > 1 :
            print("use multiprocessing on %d cpus"%threads)
        else :
            print("use single cpu")
        if covfunc == "drw":
            tag = ""
        else:
            tag = ".kepler"
        # source variability
        trufile   = "dat/trulc.dat" + tag
        # observed continuum light curve w/ seasonal gap
        confile   = "dat/loopdeloop_con.dat" + tag
        # observed continuum+y light curve w/ seasonal gap
        topfile   = "dat/loopdeloop_con_y.dat" + tag
        # observed continuum+y+z light curve w/ seasonal gap
        doufile   = "dat/loopdeloop_con_y_z.dat" + tag
        # observed continuum band+y band light curve w/out seasonal gap
        phofile   = "dat/loopdeloop_con_yb.dat" + tag
        # file for storing MCMC chains
        confchain = "dat/chain0.dat" + tag
        topfchain = "dat/chain1.dat" + tag
        doufchain = "dat/chain2.dat" + tag
        phofchain = "dat/chain3.dat" + tag

    # generate truth drw signal
    zydata  = getTrue(trufile, set_plot=set_plot, mode=mode, covfunc=covfunc)

    # generate mock light curves
    getMock(zydata, confile, topfile, doufile, phofile, set_plot=set_plot, mode=mode)

    # fit continuum
    conthpd = fitCon(confile, confchain, names=names[0:1], threads=threads, set_plot=set_plot, mode=mode)

    # fit tophat
    tophpd = fitLag(topfile, topfchain, conthpd, names=names[0:2], threads=threads, set_plot=set_plot, mode=mode)

    # fit douhat
    douhpd = fitLag(doufile, doufchain, conthpd, names=names[0:3], threads=threads,
            nwalkers=150, nburn=150, nchain=150,set_plot=set_plot, mode=mode)

    # show fit
    showfit(douhpd, doufile, names=names[0:3], set_plot=set_plot, mode=mode)

    # fit pmap
    phohpd = fitPmap(phofile, phofchain, conthpd, names=[names[0], names[3]], lagrange=[0, 150], lagbinsize=0.2, threads=threads,
            nwalkers=200, nburn=200, nchain=200, set_plot=set_plot, mode=mode)

if __name__ == "__main__":
    import sys
    print "demo test/show/run"
    mode = sys.argv[1]
    try:
        covfunc = sys.argv[2]
    except IndexError:
        covfunc = "drw"
    demo(mode, covfunc=covfunc)
