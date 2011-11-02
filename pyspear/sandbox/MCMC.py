from sampler import PowExpSampler
from pylab import *
import numpy as np

#jdata, mdata, edata = np.genfromtxt("lc.dat", unpack=True)
jdata, mdata, edata = np.genfromtxt("dat/t1000n0_6.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s4n0.5.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s3n0.5.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s3n0.5e0.05.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t10s2n0.5eobs.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t10s2n0.5eobsecov0.1.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("mock_t500s0.07n0.6ecov0.1.dat", unpack=True)

powexp = PowExpSampler('pes', jdata, mdata, edata)

iter=5000
thin=1
burn=500
powexp.sample(iter=iter,burn=burn,thin=thin)

#for i in xrange(100):
#    iter=200
#    thin=1
#    burn=0
#    powexp.sample(iter=iter,burn=burn,thin=thin)
#    powexp.db.commit()

close('all')
powexp.plot_traces()
powexp.plot_PES()
powexp.plot_post()
