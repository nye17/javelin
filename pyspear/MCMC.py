from sampler import PowExpSampler
from pylab import *
import numpy as np

#jdata, mdata, edata = np.genfromtxt("lc.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s4n0.5.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s3n0.5.dat", unpack=True)
#jdata, mdata, edata = np.genfromtxt("lc_m10t50s3n0.5e0.05.dat", unpack=True)
jdata, mdata, edata = np.genfromtxt("lc_m10t10s2n0.5e0.05.dat", unpack=True)

powexp = PowExpSampler('test', jdata, mdata, edata)

# Really takes 100k
#iter=100000
#thin=iter/2000
#burn=50000
iter=20000
thin=2
burn=2000
powexp.isample(iter=iter,burn=burn,thin=thin)

close('all')
powexp.plot_traces()
powexp.plot_PES()
powexp.plot_post()
