from javelin.lcmodel import Cont_Model, Rmap_Model
from javelin.zylc import get_data
c = get_data(['g1_min.txt'])
cmod = Cont_Model(c)
cmod.do_mcmc(set_verbose=True)
cmod.show_hist(figout='conthist', figext= 'pdf')

cl = get_data(['g1_min.txt','i1_min.txt'])
clmod = Rmap_Model(cl)
clmod.do_mcmc(conthpd=cmod.hpd,laglimit=[[-30,30]],lagtobaseline=0.3,nwalkers=100,nburn=300,nchain=500,threads=8)
clmod.show_hist(figout='line', figext= 'pdf')

par_best = clmod.hpd[1,:]
javdata_best = clmod.do_pred(par_best)
javdata_best.names=cl.names
javdata_best.plot(set_pred=True,obs=cy,figout='pred', figext= 'pdf')
