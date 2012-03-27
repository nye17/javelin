#Last-modified: 26 Mar 2012 08:58:38 PM

import os
#from javelin.lcmodel import *
#from javelin.zylc import *
#from javelin.predict import *
from lcmodel import *
from zylc import *
from predict import *
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":    
#    lineardir = "dat/LINEAR/"
    lineardir = "/data/LCDATA/LINEAR"
#    lineardir = "/home/mitchell/yingzu/Dropbox/data/LCDATA/LINEAR"
    indxfile  = os.path.join(lineardir, "LINEAR_lightcurves")
    fnamelist = []
    lcnamelist= []
    f = open(indxfile, "r")
    for line in f.readlines() :
        lcnamelist.append(line.strip())
        fnamelist.append(os.path.join(lineardir, line.strip()))
    f.close()

    # test k2e
    if True :
        testfile = fnamelist[0]
        testname = lcnamelist[0]
        zydata = get_data(testfile)
        zydata.plotdt(bins=500, log=True, set_logdt=True)
        quit()
        # show light curve
        if False :
            zydata.plot()
        # fit a DRW to data
        cont_drw = Cont_Model(zydata, covfunc="drw")
        affix  = "_data_drw" 
        fchain = os.path.join(lineardir, "k2echain_"+testname+affix)
        flogp  = os.path.join(lineardir, "k2echain_logp_"+testname+affix)
        if False :
            cont_drw.do_mcmc(set_prior=True, rank="Full",
                nwalkers=100, nburn=50, nchain=50, fburn=None,
                fchain=fchain, flogp=flogp, threads=1)
            cont_drw.get_hpd()
        else :
            cont_drw.load_chain(fchain)
#            cont_drw.show_hist(set_adaptive=True, bins=200, floor=10)
        # get sigma and tau from DRW fit to the data
        sigma, tau = np.exp(cont_drw.hpd[1,0]), np.exp(cont_drw.hpd[1,1])
        print(sigma),
        print(tau),
        # generate a mock DRW light curve
        mockfile0 = testfile+"_drwmock"
        if False :
            zymock0 = mockme(zydata, covfunc="drw", rank="Full", 
                sigma=sigma, tau=tau)
            zymock0.save(mockfile0)
        else :
            zymock0 = get_data(mockfile0)
        # show light curve
        if False :
            zymock0.plot()
        # generate a mock K2E light curve
        mockfile1 = testfile+"_k2emock_nu1"
        nu = 1.0
        print(nu)
        if False :
            zymock1 = mockme(zydata, covfunc="kepler2_exp", rank="NearlyFull", 
                sigma=sigma, tau=tau, nu=nu)
            zymock1.save(mockfile1)
        else :
            zymock1 = get_data(mockfile1)

        if False :
            zymock1.plot()

        # k2e mcmc on three light curves
#        zydatas = [zymock0,  zymock1,  zydata]
#        affixes = ["_drwmock_fitk2e","_k2emock_nu1_fitk2e","_data_fitk2e"]
        zydatas = [zymock0,  zymock1]
        affixes = ["_drwmock_fitk2e","_k2emock_nu1_fitk2e"]
        for zyda, affix in zip(zydatas, affixes) :
            cont   = Cont_Model(zyda, "kepler2_exp")
            fchain = os.path.join(lineardir, "k2echain_"+testname+affix)
            flogp  = os.path.join(lineardir, "k2echain_logp_"+testname+affix)
            if True :
                cont.do_mcmc(set_prior=True, rank="Full",
                    nwalkers=100, nburn=50, nchain=50, fburn=None,
                    fchain=fchain, flogp=flogp, threads=1)
                cont.get_hpd()
                cont.break_chain([[-3,0],None,None])
                cont.show_hist(set_adaptive=True, bins=200, floor=10)
#            else :
#                cont.load_chain(fchain)
#                cont.break_chain([None,None,[-5, 0]])
#                cont.break_chain([None,None,[0, 2]])
#                cont.break_chain([[-3,0],None,None])
#                cont.show_hist(set_adaptive=True, bins=200, floor=50)
#                cont.show_hist(set_adaptive=True, bins=200, floor=10)
#                cont.show_hist(set_adaptive=True, bins=100, floor=2)
            if False :
                # grid
                fgrid2d = os.path.join(lineardir, "k2egrid2d_"+testname+affix)
                if False :
#                    p_bst = [cont.hpd[1, 0], cont.hpd[1,1], cont.hpd[1,2]]
                    p_bst = [cont_drw.hpd[1, 0], cont_drw.hpd[1,1], 0.0]
                    fixed =  [1, 0, 0]
                    rangex = [ 0.0, 6.8]
                    dx = 0.2
                    rangey = [-5.0, 5.0]
                    dy = 0.2
                    cont.do_grid2d(p_bst, fixed, rangex, dx, rangey, dy, fgrid2d,
                            set_prior=False)
                else :
                    cont.show_logp_map(fgrid2d, set_normalize=True, vmin=-10, vmax=None,
                        set_contour=True, clevels=None, set_verbose=True)



    # test for the origin of the high-sigma solution.
    if False :
        testfile = fnamelist[0]
        testname = lcnamelist[0]
#        affix0 = "_drwmock_fitk2e"
#        mockfile0 = testfile+"_drwmock"
        affix0 = "_k2emock_nu1_fitk2e"
        mockfile0 = testfile+"_k2emock_nu1"
        fchain0 = os.path.join(lineardir, "k2echain_"+testname+affix0)
        zymock0 = get_data(mockfile0)
        cont_k2e = Cont_Model(zymock0, "kepler2_exp")
        cont_k2e.load_chain(fchain0)
#        cont_k2e.show_hist(set_adaptive=True, bins=200, floor=50)
        
        

#        p_bst = np.log([1.45, 162.0, 95.0])
        p_bst = np.log([1.45, 162.0, 0.001])
        for i, p in enumerate(cont_k2e.flatchain) :
            if p_bst[2]-0.05 < p[2] < p_bst[2]+0.05 :
                print("found")
#                print(i)
#                print(p),
#                print(np.exp(p)),
                p_bst = p
                print(cont_k2e(p_bst))
                break

        set_prior = True
        print(zymock0.cont_cad)
        print(zymock0.cont_cad_min)
        print(zymock0.cont_cad_max)
        p_bst = cont_k2e.do_map(p_bst, set_prior=set_prior)[0]
        print(np.exp(p_bst))
        print(cont_k2e(p_bst, set_retq=True, set_prior=set_prior, rank="Full"))
#        zypred0 = cont_k2e.do_pred(p_bst, rank="NearlyFull")
#        zypred0.plot(set_pred=True, obs=zymock0)

#        p_bst =  np.log([0.18, 360.0, 1e-4])
        p_bst =  np.log([0.18, 360.0, 1])
        p_bst = cont_k2e.do_map(p_bst, set_prior=set_prior)[0]
        print(np.exp(p_bst))
        print(cont_k2e(p_bst, set_retq=True, set_prior=set_prior, rank="Full"))
#        zypred0 = cont_k2e.do_pred(p_bst, rank="NearlyFull")
#        zypred0.plot(set_pred=True, obs=zymock0)



    if False :
        for lcname, fname in zip(lcnamelist, fnamelist) :
            zydata = get_data(fname, names=[lcname,])
            if False :
                zydata.plot()
            cont   = Cont_Model(zydata, "kepler2_exp")
            fchain = os.path.join(lineardir, "k2echain_"+lcname)
            flogp  = os.path.join(lineardir, "k2echain_logp_"+lcname)
            if False :
                cont.do_mcmc(set_prior=True, rank="Full",
                    nwalkers=100, nburn=50, nchain=50, fburn=None,
                    fchain=fchain, flogp=flogp, threads=1)
                cont.get_hpd()
            else :
                cont.load_chain(fchain)
    #            cont.show_hist(set_adaptive=True, bins=200, floor=40)
            fgrid2d = os.path.join(lineardir, "k2egrid2d_"+lcname)
            if True :
                p_bst = [cont.hpd[1, 0], cont.hpd[1,1], cont.hpd[1,2]]
                fixed =  [1, 0, 0]
                rangex = [ 0.0, 7.0]
                dx = 0.2
                rangey = [-6.0, 6.0]
                dy = 0.2
                cont.do_grid2d(p_bst, fixed, rangex, dx, rangey, dy, fgrid2d,
                        set_prior=False)
    #        else :
    #            cont.show_logp_map(fgrid2d, vmin=-10)
            # only do it once for testing purpose
            break
        
