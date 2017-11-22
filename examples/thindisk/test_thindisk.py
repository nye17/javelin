# 22 Nov 2017 17:14:12

from javelin.lcmodel import Cont_Model, Disk_Model
from javelin.zylc import get_data
import os.path

data0 = get_data(["driver.dat",], names=["Driver",])
data0.plot()
cont = Cont_Model(data0, "drw")
if os.path.isfile("driver_chain.dat"):
    print ("cont model chain exists")
    cont.load_chain('driver_chain.dat')
    cont.show_hist(bins=100)
else:
    cont.do_mcmc(nwalkers=100, nburn=100, nchain=200, fburn=None, fchain="driver_chain.dat", threads=1)

conthpd = cont.hpd
print conthpd

data1 = get_data(["driver.dat", "wave2.dat", "wave3.dat", "wave4.dat"], names=["Driver", "Wave 2", "Wave 3", "Wave 4"])
data1.plot()
disk1 = Disk_Model(data1, effwave=[2000., 4000., 5000., 8000.])
if os.path.isfile("thin_disk_chain.dat"):
    print ("disk model chain exists")
    disk1.load_chain('thin_disk_chain.dat')
    disk1.show_hist()
else:
    disk1.do_mcmc(conthpd=conthpd, nwalkers=100, nburn=100, nchain=500, threads=1, fchain="thin_disk_chain.dat", flogp="thin_disk_flogp.dat", fburn="thin_disk_burn.dat")

