from prh import PRH, func
from zylc import get_data
import numpy as np
from emcee import EnsembleSampler



if __name__ == "__main__":    
    lcfile = "dat/loopdeloop_con.dat"
    zylc   = get_data(lcfile)
    cont   = PRH(zylc)
    nwalkers = 100
    p0 = np.random.rand(nwalkers*2).reshape(nwalkers, 2)
    p0[:,0] = np.abs(p0[:,0]) - 0.5
    p0[:,1] = np.abs(p0[:,1]) + 1.0
#    sampler = EnsembleSampler(nwalkers, 2, cont, threads=2)
    sampler = EnsembleSampler(nwalkers, 2, func, args=[cont,], threads=2)
#    sampler = EnsembleSampler(nwalkers, 2, cont, threads=1)

    pos, prob, state = sampler.run_mcmc(p0, 100)
    np.savetxt("burn.out", sampler.flatchain)
    print("burn-in finished\n")
    sampler.reset()
    sampler.run_mcmc(pos, 100, rstate0=state)
    af = sampler.acceptance_fraction
    print(af)
    np.savetxt("test.out", sampler.flatchain)
    plt.hist(np.exp(sampler.flatchain[:,0]), 100)
    plt.show()
    plt.hist(np.exp(sampler.flatchain[:,1]), 100)
    plt.show()

