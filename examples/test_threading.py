from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model
import pickle

def fitCon(confile, confchain, names=None, threads=4, set_plot=False, nwalkers=200, nburn=50, nchain=200):
    """ fit the continuum model.
    """
    zydata = get_data(confile, names=names)
    cont = Cont_Model(zydata, "drw")
    # s = pickle.dumps(cont)
    cont.do_mcmc(nwalkers=nwalkers, nburn=nburn, nchain=nchain, fburn=None, fchain=confchain, threads=threads)
    if set_plot :
        cont.show_hist(bins=100, figout="mcmc0", figext=None)
    return(cont.hpd)

def fitLag(linfile, linfchain, conthpd, names=None, lagrange=[50, 300], lagbinsize=1, threads=1, set_plot=False, nwalkers=100, nburn=100, nchain=100) :
    """ fit the Rmap model.
    """
    zydata = get_data(linfile, names=names)
    rmap   = Rmap_Model(zydata)
    laglimit = [[0.0, 400.0],]*(zydata.nlc-1)
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
        rmap.show_hist(bins=100, lagbinsize=lagbinsize, figout=figout, figext=None)
    return(rmap.hpd)


if __name__ == "__main__":
    confile = "dat/loopdeloop_con.dat"
    topfile   = "dat/loopdeloop_con_y.dat"
    confchain = "dat/chain0_threading.dat"
    topfchain = "dat/chain1_threading.dat"
    names = ["Continuum", "Yelm", "Zing", "YelmBand"]
    threads = 1
    # conthpd = fitCon(confile, confchain, names=names[0:1], threads=threads, set_plot=False)
    conthpd = None
    tophpd = fitLag(topfile, topfchain, conthpd, names=names[0:2], threads=threads, set_plot=True)
