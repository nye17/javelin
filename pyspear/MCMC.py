from sampler import PowExpSampler
from pylab import *
import numpy as np

jdata, mdata, edata = np.genfromtxt("lc.dat", unpack=True)

powexp = PowExpSampler('test', jdata, mdata, edata)

# Really takes 100k
#iter=100000
#thin=iter/2000
#burn=50000
iter=5000
thin=1
burn=500
powexp.isample(iter=iter,burn=burn,thin=thin)

close('all')
powexp.plot_traces()
powexp.plot_PES()
powexp.plot_post()
