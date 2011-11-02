
import numpy as np
import matplotlib.pyplot as plt


def plot_decomp(ttau, tnu):
#    record = "dat/OGLE_example/pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau.dat"
    record  = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau.dat"
    figname = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau.pdf"
    print(record)
    print(figname)
#    0.0700     1.0000     0.1925   998.0161  -234.9490  1237.2307    -4.2657     0.0002
    sig, tau, nu, loglike, chi2, complex, drift, q = np.genfromtxt(record, unpack=True)
    pmax = np.max(loglike)
    loglike -= pmax
    pmax = np.max(chi2)
    chi2    -= pmax
    pmax = np.max(complex)
    complex -= pmax
    pmax = np.max(drift)
    drift   -= pmax

    indx = np.argmax(loglike)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax1.plot(tau, loglike, "k-", label="marginal likelihood")
    ax1.plot(tau, chi2, "r--", label="data fit")
    ax1.plot(tau, complex, "g--", label="minus complexity penalty")
    ax1.plot(tau, drift, "b--", label="minus q variance penalty")
    ax1.axvline(ttau, ls=":", color="k", lw=2, alpha=0.6)
    ax1.axhline(0.0, ls=":", color="k", alpha=0.5)
    ax1.axvline(tau[indx], ls="-", color="k", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\tau_d$")
    ax1.set_ylabel("log probability")
    ax1.set_ylim(-20, 2)
    leg  = ax1.legend(loc=4)
    leg.get_frame().set_alpha(0.5)
   
    plt.show()
#    fig.savefig(figname)



if __name__ == "__main__":    
    plot_decomp(20.0, 0.6)
    plot_decomp(20.0, 1.0)
    plot_decomp(20.0, 1.4)
    plot_decomp(100.0, 0.6)
    plot_decomp(100.0, 1.0)
    plot_decomp(100.0, 1.4)
    plot_decomp(600.0, 0.6)
    plot_decomp(600.0, 1.0)
    plot_decomp(600.0, 1.4)
    plot_decomp(1000.0, 0.6)
    plot_decomp(1000.0, 1.0)
    plot_decomp(1000.0, 1.4)



