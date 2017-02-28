from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model
from javelin.lcmodel import Rmap_Model


for i in xrange(50):
	c = get_data(["dat/continuum.dat"])
	cmod = Cont_Model(c)
	cmod.do_mcmc(nwalkers=200, nburn=100, nchain=200, threads = 8)
	a = cmod.get_hpd()
	javdata4 = get_data(["dat/continuum.dat", "dat/yelm.dat", "dat/zing.dat"], names=["Continuum", "Yelm", "Zing"])
	rmap2 = Rmap_Model(javdata4)
	rmap2.do_mcmc(conthpd=a, nwalkers=200, nburn=100, nchain=200, threads = 8)

	del rmap2
	del c
	print i
