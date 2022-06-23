from javelin.lcmodel import Cont_Model, Rmap_Model
from javelin.zylc import get_data
c = get_data(['g1_min.txt'])
cmod = Cont_Model(c)
cmod.do_mcmc(set_verbose=True, nwalkers=100, nburn=50, nchain=50, fchain='contchain.dat', threads=1)
# cmod.load_chain('contchain.dat'); cmod.get_hpd()
cmod.show_hist(figout='conthist', figext= 'pdf')


cl = get_data(['g1_min.txt','i1_min.txt'])
clmod = Rmap_Model(cl)
clmod.do_mcmc(conthpd=cmod.hpd,laglimit=[[-30,30]], lagtobaseline=0.3,
        nwalkers=100,nburn=100,nchain=100,threads=10, fchain='linechain.dat')
# clmod.load_chain('linechain.dat'); clmod.get_hpd()
clmod.show_hist(figout='linehist', figext= 'pdf')

par_best = clmod.hpd[1,:]
javdata_best = clmod.do_pred(par_best)
javdata_best.names=cl.names
javdata_best.plot(set_pred=True,obs=cl,figout='pred', figext= 'pdf')
