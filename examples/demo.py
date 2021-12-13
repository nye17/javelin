#Last-modified: 08 Dec 2013 15:54:47
import numpy as np
import matplotlib.pyplot as plt
from javelin.predict import PredictSignal, PredictRmap, generateLine, generateError, PredictSpear
from javelin.lcio import *
from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model, DPmap_Model

""" Tests from scratch.
"""

#************** PLEASE DO NOT EDIT THIS PART*****
# show figures interactively
figext = 'pdf'
# names of the true light curves
names  = ["Continuum", "Yelm", "Zing", "YelmBand", "YelmZingBand"]
# dense sampling of the underlying signal
jdense = np.linspace(0.0, 2000.0, 2000)
# DRW parameters
sigma, tau = (3.00, 400.0)
# tau_cut
tau_cut = 7.0
# line parameters
lagy, widy, scaley = (100.0, 2.0, 0.5)
lagz, widz, scalez = (250.0, 4.0, 0.5)
lags   = [0.0,   lagy,   lagz]
wids   = [0.0,   widy,   widz]
scales = [1.0, scaley, scalez]
llags   = [   lagy,   lagz]
lwids   = [   widy,   widz]
lscales = [ scaley, scalez]
lcmeans= [10.0, 5.0, 5.0]
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
        if i == 2:
            # special yelm prediction for YelmZing at the observed epochs of the ZingLine
            jl_y, sl_y = generateLine(jdense, sdense, lags[1], wids[1], scales[1], mc_mean=0.0, ml_mean=0.0)
            imin_y = np.searchsorted(jl_y, jmin)
            imax_y = np.searchsorted(jl_y, jmax)
            jdense_yz = jl_y[imin_y: imax_y]
            edense_yz = np.zeros_like(jdense_yz)
            sdense_yz = sl_y[imin_y: imax_y]+lcmeans[1]
            dplc = [jdense_yz, sdense_yz + zylist[2][2], edense_yz]
    # this is for handling the prediction for YelmBand.
    # combine into a single LightCurve
    zylist.append(phlc)
    # this is for handling the prediction for YelmZing.
    zylist.append(dplc)
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
    zylistnew.append(phlc)
    # this is for handling the prediction for YelmZingBand.
    dplc = [jdense, mdense, edense]
    dplc[1] = zylistnew[1][1] + zylistnew[2][1]
    # combine into a single LightCurve
    zylistnew.append(dplc)
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

def getMock(zydata, confile, topfile, doufile, phofile, dphfile, set_plot=False, mode="test") :
    """ downsample the truth to get more realistic light curves
    """
    if mode == "test" :
        return(None)
    else :
        _c, _y, _z, _yb, _yzb = zydata.split()
        _zydata = _c + _y + _z
        zydata_dou = True2Mock(_zydata, sparse=[8, 8, 8], errfac=[0.01, 0.01, 0.01], hasgap=[False, False, False], errcov=0.0)
        zylclist_top = zydata_dou.zylclist[:2]
        zydata_top = LightCurve(zylclist_top, names=names[0:2])
        _zydata = _c + _yb
        zydata_pho = True2Mock(_zydata, sparse=[8, 8], errfac=[0.01, 0.01], hasgap=[False, False], errcov=0.0)
        _zydata = _c + _yzb
        zydata_dph = True2Mock(_zydata, sparse=[8, 8], errfac=[0.01, 0.01], hasgap=[False, False], errcov=0.0)
        if mode == "run" :
            confile = ".".join([confile, "myrun"])
            doufile = ".".join([doufile, "myrun"])
            topfile = ".".join([topfile, "myrun"])
            phofile = ".".join([phofile, "myrun"])
            dphfile = ".".join([dphfile, "myrun"])
            zydata_dou.save_continuum(confile)
            zydata_dou.save(doufile)
            zydata_top.save(topfile)
            zydata_pho.save(phofile)
            zydata_dph.save(dphfile)
    if set_plot :
        print("plot mock light curves for continuum, yelm, zing, yelm band, and yelm+zing band lines")
        _c, _yb = zydata_pho.split()
        _c, _yzb = zydata_dph.split()
        zymock = zydata_dou + _yb + _yzb
        zymock.plot(figout="mocklc", figext=figext)

def fitCon(confile, confchain, names=None, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100, mode="test") :
    """ fit the continuum model.
    """
    if mode == "run" :
        confile = ".".join([confile, "myrun"])
    zydata = get_data(confile, names=names)
    cont   = Cont_Model(zydata, "drw")
    if mode == "test" :
        print(cont([np.log(sigma), np.log(tau)], set_retq=True))
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
            print(rmap([np.log(sigma), np.log(tau), lagy, widy, scaley ]))
        elif zydata.nlc == 3 :
            print(rmap([np.log(sigma), np.log(tau), lagy, widy, scaley, lagz, widz, scalez]))
        return(None)
    elif mode == "show" :
        rmap.load_chain(linfchain, set_verbose=False)
    elif mode == "run" :
        laglimit = [[0.0, tau],]*(zydata.nlc-1)
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

def fitPmap(phofile, phofchain, conthpd, names=None, lagrange=[50, 300], lagbinsize=1, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100,mode="test", fixed=None, p_fix=None) :
    """ fit the Pmap model.
    """
    if mode == "run" :
        phofile = ".".join([phofile, "myrun"])
    zydata = get_data(phofile, names=names)
    pmap   = Pmap_Model(zydata)
    if mode == "test" :
        print(pmap([np.log(sigma), np.log(tau), lagy, widy, scaley,  1.0]))
        return(None)
    elif mode == "show" :
        pmap.load_chain(phofchain, set_verbose=False)
    elif mode == "run" :
        # laglimit = [[50.0, 130.0]] # XXX here we want to avoid 180 day limit.
        laglimit = [lagrange,] # XXX here we want to avoid 180 day limit.
        widlimit = [[0, 7.0]] # XXX here we want to avoid long smoothing width
        phofchain = ".".join([phofchain, "myrun"])
        pmap.do_mcmc(conthpd=conthpd, lagtobaseline=0.5, laglimit=laglimit,
                widlimit=widlimit, nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=phofchain, threads=threads, fixed=fixed, p_fix=p_fix)
    if set_plot :
        pmap.break_chain([lagrange,])
        pmap.get_hpd()
        pmap.show_hist(bins=100, lagbinsize=lagbinsize, figout="mcmc3", figext=figext)
    return(pmap.hpd)

def fitDPmap(dphfile, dphfchain, conthpd, names=None, lagrange=[-50, 300], lagbinsize=1, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100,mode="test", fixed=None, p_fix=None) :
    """ fit the DPmap model.
    """
    if mode == "run" :
        dphfile = ".".join([dphfile, "myrun"])
    zydata = get_data(dphfile, names=names)
    dpmap   = DPmap_Model(zydata)
    if mode == "test" :
        print(dpmap([np.log(sigma), np.log(tau), lagz, widz, scalez, lagy, widy, scaley]))
        return(None)
    elif mode == "show" :
        dpmap.load_chain(dphfchain, set_verbose=False)
    elif mode == "run" :
        laglimit = [lagrange,lagrange]
        # laglimit = [lagrange,[-50, 300]] # FIXME temperary fix
        widlimit = [[0, 7.0], [0, 7.0]] # XXX here we want to avoid long smoothing width
        dphfchain = ".".join([dphfchain, "myrun"])
        dpmap.do_mcmc(conthpd=conthpd, lagtobaseline=0.5, laglimit=laglimit,
                widlimit=widlimit, nwalkers=nwalkers, nburn=nburn, nchain=nchain,
                fburn=None, fchain=dphfchain, threads=threads, fixed=fixed, p_fix=p_fix)
    if set_plot :
        # dpmap.break_chain([lagrange,[-50, 300]])
        dpmap.break_chain([lagrange,lagrange])
        dpmaphpd = dpmap.get_hpd()
        dpmap.show_hist(bins=100, lagbinsize=lagbinsize, figout="mcmc4", figext=figext)
    # if True:
        # zypred = dpmap.do_pred(dpmap.hpd[1,:])
        # zypred.names = names
        # zypred.plot(set_pred=True, obs=zydata, figout="prediction", figext=figext)
    return(dpmap.hpd)

def showfit(linhpd, linfile, names=None, set_plot=False, mode="test", model='Rmap') :
    if mode == "run" :
        linfile = ".".join([linfile, "myrun"])
    print linfile
    zydata = get_data(linfile, names=names)
    if model == 'Rmap' or model == 'Rmap2':
        rmap = Rmap_Model(zydata)
    elif model == 'Pmap':
        rmap = Pmap_Model(zydata)
    elif model == 'DPmap':
        rmap = DPmap_Model(zydata)
    if mode == "test" :
        return(None)
    else :
        zypred = rmap.do_pred(linhpd[1,:])
        zypred.names = names
        print names
    if set_plot :
        zypred.plot(set_pred=True, obs=zydata, figout="prediction_" + model, figext=figext)

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
            # if True:
                threads = multiprocessing.cpu_count()
                if threads > 40:
                    threads = 40
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
        # observed continuum band+yz band light curve w/out seasonal gap
        dphfile   = "dat/loopdeloop_con_yzb.dat" + tag
        # file for storing MCMC chains
        confchain = "dat/chain0.dat" + tag
        topfchain = "dat/chain1.dat" + tag
        doufchain = "dat/chain2.dat" + tag
        phofchain = "dat/chain3.dat" + tag
        dphfchain = "dat/chain4.dat" + tag

    # generate truth drw signal
    zydata  = getTrue(trufile, set_plot=set_plot, mode=mode, covfunc=covfunc)

    # generate mock light curves
    getMock(zydata, confile, topfile, doufile, phofile, dphfile, set_plot=set_plot, mode=mode)

    # fit continuum
    conthpd = fitCon(confile, confchain, names=names[0:1], threads=threads, set_plot=set_plot, mode=mode)
    if conthpd is None:
        conthpd = np.array([[0.934, 5.536], [1.139, 5.953], [1.452, 6.568]])
    else:
        print(conthpd)

    # fit tophat
    tophpd = fitLag(topfile, topfchain, conthpd, names=names[0:2], threads=threads, set_plot=set_plot, mode=mode)
    if tophpd is None:
        tophpd = np.array([[0.542, 4.886, 99.704, 0.308, 0.491], [  0.661, 5.085, 100.015,  0.76, 0.497], [  0.793,  5.369, 100.34, 1.668,  0.503]])
    else:
        print(tophpd)

    showfit(tophpd, topfile, names=names[0:2], set_plot=set_plot, mode=mode, model='Rmap')

    # fit douhat
    douhpd = fitLag(doufile, doufchain, conthpd, names=names[0:3], threads=threads, nwalkers=100, nburn=100, nchain=100,set_plot=set_plot, mode=mode)
    if douhpd is None:
        douhpd = np.array([[  1.132,  5.905,  99.591,  0.405,  0.502, 246.547,   0.507,   0.505],
                           [  1.176,  6.044, 100.411,  0.494,  0.518, 250.251,   0.572,   0.521],
                           [  1.355,  6.127, 107.987,  0.555,  0.579, 251.629,   0.671,   0.594]])
    else:
        print(douhpd)

    # show fit
    showfit(douhpd, doufile, names=names[0:3], set_plot=set_plot, mode=mode, model='Rmap2')

    # fit pmap
    phohpd = fitPmap(phofile, phofchain, conthpd, names=[names[0], names[3]], lagrange=[-50, 300], lagbinsize=0.2, threads=threads, nwalkers=100, nburn=100, nchain=100, set_plot=set_plot, mode=mode)
    if phohpd is None:
        phohpd =np.array([[  1.098,   5.798, 100.218,   1.331,   0.504,   0.971],
                          [  1.214,   6.061, 100.693,   3.156,   0.512,   0.98 ],
                          [  1.396,   6.318, 101.069,   4.861,   0.521,   0.989]])
    else:
        print(phohpd)
    showfit(phohpd, phofile, names=[names[0], names[3]], set_plot=set_plot, mode=mode, model='Pmap')

    # fit dpmap
    dphhpd = fitDPmap(dphfile, dphfchain, conthpd, names=[names[0], names[4]], lagrange=[50, 300], lagbinsize=0.2, threads=threads, nwalkers=100, nburn=100, nchain=100, set_plot=set_plot, mode=mode)
    if dphhpd is None:
        dphhpd = np.array([[  0.883,   4.77 , 173.676,   1.632,   0.339,  99.206,   2.249,   0.495],
                           [  0.958,   5.334, 249.566,   2.981,   0.516,  99.897,   3.778,   0.514],
                           [  1.234,   5.593, 250.5  ,   3.822,   0.538, 171.595,   4.472,   0.797]])
    else:
        print(dphhpd)
    showfit(dphhpd, dphfile, names=[names[0], names[4]], set_plot=set_plot, mode=mode, model='DPmap')

if __name__ == "__main__":
    import sys
    print "demo test/show/run"
    mode = sys.argv[1]
    try:
        covfunc = sys.argv[2]
    except IndexError:
        covfunc = "drw"
    demo(mode, covfunc=covfunc)
