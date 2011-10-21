import predict as pd
import numpy as np

jd, mag, emag = np.genfromtxt("lc3c.dat", usecols=(0,1,2), unpack=True)


#pd.predict.setup_observed_lc(jd, mag, emag)

pd.predict.jdata =jd
pd.predict.mdata =mag
pd.predict.edata =emag

jdnew = np.arange(500, 600, 1)
emagnew = np.zeros_like(jdnew)
meanemag = np.mean(emag)
emagnew = emagnew+meanemag

pd.predict.setup_desired_lc(jdnew, emagnew)

pd.predict.constrained_drw(50, 0.05)


