
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt


def posterier_mean(loglike, x):
    p = np.exp(loglike)
    u = trapz(p*x, x)
    d = trapz(p, x)
    return(u/d)
    

def plot_decomp(ttau, tnu, tsig, tmean):

    lcfile  = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".dat"
    j, m, e = np.genfromtxt(lcfile, unpack=True)
    lcmean = np.mean(m)
    record     = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_hires.dat"
    recorddrw  = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_DRW.dat"
    figname    = "pow_exp_T"+str(ttau)+"_N"+str(tnu)+".testtau_hires.pdf"
    print(lcfile)
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

    taudrw, loglikedrw = np.genfromtxt(recorddrw, usecols=(1,3), unpack=True)
    pmaxdrw = np.max(loglikedrw)
    loglikedrw -= pmaxdrw
    indxdrw = np.argmax(loglikedrw)


    # get the posterior mean
    tau_post = np.power(10.0, posterier_mean(loglike, np.log10(tau)))

    fig = plt.figure(figsize=(12, 12))
    parset_in = {"\sigma": tsig,
                 "\\tau_d": ttau,
                 "\\nu": tnu,
                 "\\bar{m}": tmean
                 }
    title_in = '    '.join('$%s$=%s' % (k,format(v, "6.3f")) for k,v in parset_in.items())

    parset_out = {"\sigma": sig[indx],
                 "\\tau_d": tau[indx],
                 "\\nu": nu[indx],
                 "\\bar{m}": lcmean+q[indx]
                 }
    title_out = '    '.join('$%s$=%s' % (k,format(v, "6.3f")) for k,v in parset_out.items())


    ax1 = fig.add_axes([0.10, 0.08, 0.87, 0.25])
    ax1.plot(tau, loglike, "y-", lw=2, label="marginal likelihood")
    ax1.plot(tau, chi2, "r--", label="data fit")
    ax1.plot(tau, complex, "g--", label="minus complexity penalty")
    ax1.plot(tau, drift, "b--", label="minus q variance penalty")
    ax1.plot(tau, loglikedrw, "m-", lw=1, label="DRW likelihood")
    ax1.axhline(0.0, ls="-", color="k", lw=1, alpha=0.6)
    ax1.axvline(ttau, ls="-", color="k", lw=3, alpha=0.3)
    ax1.axvline(tau[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax1.axvline(taudrw[indxdrw], ls="-", color="m", lw=2, alpha=0.6)
    ax1.axvline(tau_post, ls="-", color="purple", lw=1, alpha=0.9)
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\tau_d$")
    ax1.set_ylabel("log probability")
    ax1.set_xlim(1, 12000)
    ax1.set_ylim(-20, 3)
    leg  = ax1.legend(loc=4)
    leg.get_frame().set_alpha(0.5)
   

    ax2 = fig.add_axes([0.10, 0.40, 0.25, 0.25])
    ax2.plot(tau, sig, "r-", lw=2)
    ax2.axhline(tsig, ls="-", color="k", lw=2, alpha=0.6)
    ax2.axvline(ttau, ls="-", color="k", lw=2, alpha=0.6)
    ax2.axhline(sig[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax2.axvline(tau[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("$\\tau_d$")
    ax2.set_ylabel("$\\sigma$")
    ax2.set_xlim(1, 12000)
    ax2.set_ylim(0.02, 0.5)
    yticks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([format(k, "2g") for k in yticks])

    ax3 = fig.add_axes([0.41, 0.40, 0.25, 0.25])
    ax3.plot(tau, nu, "r-", lw=2)
    ax3.axhline(tnu, ls="-", color="k", lw=2, alpha=0.6)
    ax3.axvline(ttau, ls="-", color="k", lw=2, alpha=0.6)
    ax3.axhline(nu[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax3.axvline(tau[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax3.set_xscale("log")
    ax3.set_xlabel("$\\tau_d$")
    ax3.set_ylabel("$\\nu$")
    ax3.set_xlim(1, 12000)
    ax3.set_ylim(0, 2)

    ax4 = fig.add_axes([0.72, 0.40, 0.25, 0.25])
    ax4.plot(tau, q, "r-", lw=2)
    ax4.axhline(tmean-lcmean, ls="-", color="k", lw=2, alpha=0.6)
    ax4.axvline(ttau, ls="-", color="k", lw=2, alpha=0.6)
    ax4.axhline(q[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax4.axvline(tau[indx], ls="-", color="y", lw=2, alpha=0.6)
    ax4.set_xscale("log")
#    ax4.set_yscale("log")
    ax4.set_xlabel("$\\tau_d$")
    ax4.set_ylabel("$q$")
    ax4.set_xlim(1, 12000)
    ax4.set_ylim(-0.1, 0.1)
    ax4.ticklabel_format(style='sci', scilimits=(0,0), axis='y') 

    ax5 = fig.add_axes([0.10, 0.71, 0.87, 0.25])
    ax5.errorbar(j, m, yerr=e, linewidth=0.5, marker="o", ls="None", mfc="r", mec="k", color="k", alpha=0.5, markersize=5)
    ax5.axhline(tmean, ls="-", color="k", lw=1, alpha=1.0)
    qmin = q.min()
    qmax = q.max()
    ax5.axhspan(lcmean+qmin, lcmean+qmax,   color="y", alpha=0.6)
    ax5.set_xlabel("JD")
    ax5.set_ylabel("Mag")

    ax5.text(.5, .95, "true parameters: "+title_in, horizontalalignment='center', va="top", transform = ax5.transAxes)
    ax1.text(.5, .95, "MLE parameters: "+title_out, horizontalalignment='center', va="top", transform = ax1.transAxes)
    ax1.text(.05, .05, "posterier mean of $\\tau$: "+str(tau_post), horizontalalignment='left', va="bottom", transform = ax1.transAxes)

#    plt.show()
    fig.savefig(figname)



if __name__ == "__main__":    
    plot_decomp(20.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(20.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(40.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(40.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(60.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(60.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(80.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(80.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(100.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(100.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(200.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(200.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(400.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(400.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(600.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(600.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(800.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(800.0, 1.8, 0.0819952166662, 18.6782617587)

    plot_decomp(1000.0, 0.2, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 0.4, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 0.6, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 0.8, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 1.0, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 1.2, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 1.4, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 1.6, 0.0819952166662, 18.6782617587)
    plot_decomp(1000.0, 1.8, 0.0819952166662, 18.6782617587)


