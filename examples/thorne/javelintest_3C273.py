from __future__ import absolute_import
from __future__ import print_function
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model, Disk_Model, DPmap_Model
from javelin.zylc import get_data
import os.path

c = get_data(["3C273_Bband.dat"])
cmod = Cont_Model(c, "drw")
cmod.do_mcmc()
cmod.show_hist(figout='3C273_Bband_sig-tau_distributions', figext= 'pdf')

cy = get_data(["3C273_Bband.dat", "3C273_Gband.dat"], names=["B band", "G band"])
cymod = Rmap_Model(cy)
cymod.do_mcmc(conthpd=cmod.hpd, laglimit=[[1,10]], fchain='3C273_BGchain.dat')

cymod.show_hist(figout='3C273_BGdistributions', figext= 'pdf')

par_best = cymod.hpd[1,:]
javdata_best = cymod.do_pred(par_best)
javdata_best.names=cy.names
javdata_best.plot(set_pred=True,obs=cy,figout='3C273_BGgraphs', figext= 'pdf')

