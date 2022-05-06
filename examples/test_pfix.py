from __future__ import absolute_import
from __future__ import print_function
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model
from javelin.lcmodel import Rmap_Model
import numpy as np


c = get_data(["dat/continuum.dat"])
cmod = Cont_Model(c)
cmod.do_mcmc(nwalkers=100, nburn=100, nchain=100, threads = 1)
cmod.show_hist()
print(cmod.hpd)
javdata4 = get_data(["dat/continuum.dat", "dat/yelm.dat"], names=["Continuum", "Yelm"])
rmap2 = Rmap_Model(javdata4)
rmap2.do_mcmc(conthpd=cmod.hpd, nwalkers=100, nburn=100, nchain=100, threads = 1,
              laglimit=[[10, 200],], fixed=[1, 0, 1, 0, 1], p_fix=[np.log(0.1), np.log(400), 10, 2.0, 5.0])
rmap2.show_hist()
print(rmap2.hpd)
