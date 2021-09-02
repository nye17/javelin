# 02 Sep 2021 01:56:41

from javelin.lcmodel import Cont_Model, DPmap_Model
from javelin.zylc import get_data
import os.path

data0 = get_data(['Mrk509Gband.txt'], names=["G-band",])
data0.plot(figout='lc_g', figext='pdf')
cont = Cont_Model(data0, "drw")
if os.path.isfile("cont_chain.dat"):
    print ("cont model chain exists")
    cont.load_chain('cont_chain.dat')
    # cont.show_hist(bins=100, figext='pdf')
else:
    cont.do_mcmc(nwalkers=100, nburn=100, nchain=100, fburn=None, fchain="cont_chain.dat", threads=8)

conthpd = cont.hpd
print conthpd

data1 = get_data(["Mrk509Gband_short.txt", "Mrk509Hband_normalized.txt"], names=["G-band", "H-band"])
data1.plot(figout='lc_gh', figext='pdf')
# quit()
dust = DPmap_Model(data1)
if os.path.isfile("dpmap_chain.dat"):
    print ("double echo model chain exists")
    dust.load_chain('dpmap_chain.dat')
    dust.show_hist(figout='mcmc_all', figext='pdf')
    # dust.break_chain([[280, 310],[180, 230]])
    dust.break_chain([[280, 310],[230, 270]])
    # dust.break_chain([[280, 310],[0, 400]])
    dust.get_hpd()
    dust.show_hist(figout='mcmc', figext='pdf')
    if True:
        zypred = dust.do_pred(dust.hpd[1,:])
        zypred.names = data1.names
        zypred.plot(set_pred=True, obs=data1, figout="prediction", figext='pdf')
else:
    laglimit = [[0, 400], [0, 400]]
    dust.do_mcmc(conthpd=conthpd, nwalkers=300, nburn=300, nchain=300, threads=8, lagtobaseline=0.5, laglimit=laglimit, fchain="dpmap_chain.dat", flogp="dpmap_flogp.dat", fburn="dpmap_burn.dat")

