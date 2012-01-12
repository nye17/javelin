#Last-modified: 11 Jan 2012 04:16:03 PM
import numpy as np
import matplotlib.pyplot as plt



def mk_dehnen(const, slow, tau, alpha, beta):
    """ slow is determined by the light curve duration
    """
    def dehnen(s):
        if s <= slow : 
            _val = const/((slow/tau)**alpha*(1.+slow/tau)**(beta-alpha))
        else :
            _val = const/((s/tau)**alpha*(1.+s/tau)**(beta-alpha)) 
        return(_val)
    return(dehnen)

def show_dehnen(alpha=2, beta=3):
    dehnen = mk_dehnen(const=1.0, slow=0.001, tau=1.0, alpha=alpha, beta=beta)
    s = np.power(10.0, np.arange(-5, 2, 0.01))
    p = np.zeros_like(s)
    for i, ss in enumerate(s):
        p[i] = dehnen(ss)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(s, p, "k-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e7)
    plt.show()

def mk_kepler(gamma, tcut, tau, sigma):
    # tcut is not a ratio here
    def kepler(t):
        tau2 = tau
        sigma2drw = sigma**2*np.exp(-np.abs(tcut/tau2)**gamma+np.abs(tcut/tau))
        if np.abs(t) >= tcut :
            _val = sigma2drw*np.exp(-np.abs(t/tau))
        else :
            _val = sigma**2*np.exp(-np.abs(t/tau2)**gamma)
        return(_val)
    return(kepler)

def show_kepler(gamma = 2.0, tcut=0.2, tau=1.0, sigma=1.):
    kepler = mk_kepler(gamma, tcut, tau, sigma)
    tau2 = tau
    sigma2drw = sigma**2*np.exp(-np.abs(tcut/tau2)**gamma+np.abs(tcut/tau))
    t = np.arange(0, 2, 0.01)
    drw = lambda z : sigma2drw*np.exp(-np.abs(z/tau))
    c = np.zeros_like(t)
    c_drw = np.zeros_like(t)
    for i, tt in enumerate(t):
        c[i] = kepler(tt)
        c_drw[i] = drw(tt) 
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.plot(t, c, "k-")
    ax.plot(t, c_drw, "r--")
    plt.show()




if __name__ == "__main__":    
    gamma = 2.0; tcut=0.5; tau=1.0; sigma=1.
#    show_kepler(gamma, tcut, tau, sigma)
    f = mk_kepler(gamma, tcut, tau, sigma)

