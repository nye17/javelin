#Last-modified: 16 Jan 2012 05:46:40 PM
import numpy as np
import pickle
import os.path

from prh import PRH
from peakdetect import peakdet1d, peakdet2d

import matplotlib.pyplot as plt


def getPlike(zydata, par, covfunc, set_verbose=False):
    """ Calculate the log marginal likelihood of data given input paramters.
    """
    if set_verbose:
        print("**************************************")
        print("Input  sigma, tau, nu")
        print(pretty_array(par))
    prh = PRH(zydata, covfunc=covfunc, sigma=par[0], tau=par[1], nu=par[2])
    out = prh.loglike_prh()
    if set_verbose:
        print("--------------------------------------")
        print("Output logL, -chi2/2, complexity, drift, [q]")
        print(pretty_array(out))
        print("**************************************")
    return(out)

def runMAP(model, set_verbose=True):
    """ Running MAP ananlysis.
    """
    M = pm.MAP(model)
    tovary = getValues(M)
    if set_verbose:
        print("**************************************")
        print("Initial sigma, tau, nu")
        print(pretty_array(tovary))
    if set_verbose:
        verbose=1
    else:
        verbose=0
    M.fit(verbose=verbose)
    tovary = getValues(M)
    if set_verbose:
        print("--------------------------------------")
        print("Bestfit sigma, tau, nu")
        print(pretty_array(tovary))
        print("**************************************")
    return(tovary)

def read_grid_tau_nu(input):
    """ read the grid file
    """ 
    print("reading from %s"%input)
    f = open(input, "r")
    dim_tau, dim_nu = [int(r) for r in f.readline().lstrip("#").split()]
    print("dim of tau: %d"%dim_tau)
    print("dim of  nu: %d"%dim_nu)
    sigma, tau, nu, loglike, chi2, complexity, drift, q = np.genfromtxt(f, unpack=True)
    f.close()
    retdict = {
               'sigma'          :        sigma.reshape(dim_tau, dim_nu),
               'tau'            :          tau.reshape(dim_tau, dim_nu),
               'nu'             :           nu.reshape(dim_tau, dim_nu),
               'loglike'        :      loglike.reshape(dim_tau, dim_nu),
               'chi2'           :         chi2.reshape(dim_tau, dim_nu),
               'complexity'     :   complexity.reshape(dim_tau, dim_nu),
               'drift'          :        drift.reshape(dim_tau, dim_nu),
               'q'              :            q.reshape(dim_tau, dim_nu),
              }
    return(retdict)

def show_loglike_map(x, y, z, ax, cax=None, 
                     set_contour=True, clevels=None,
                     vmin=None, vmax=None, xlabel=None, ylabel=None, zlabel=None, 
                     set_normalize=True, peakpos=None, cmap='jet'):
    """ Display the 2D likelihood maps, assuming xmesh and ymesh are on regular grids.
    """
    xmin,xmax,ymin,ymax = np.min(x),np.max(x),np.min(y),np.max(y)
    extent = (xmin,xmax,ymin,ymax)
    if set_normalize:
        zmax = np.max(z)
        z    = z - zmax
    if vmin is None:
        vmin = z.min()
    if vmax is None:
        vmax = z.max()
    im = ax.imshow(z, origin='lower', vmin=vmin, vmax=vmax,
                      cmap=cmap, interpolation="nearest", aspect="auto", extent=extent)
    if set_contour:
        if clevels is None:
            sigma3,sigma2,sigma1 = 11.8/2.0,6.17/2.0,2.30/2.0
            levels = (vmax-sigma1, vmax-sigma2, vmax-sigma3)
        else:
            levels = clevels
        ax.set_autoscale_on(False)
        cs = ax.contour(z,levels, hold='on',colors='k',
                          origin='lower',extent=extent)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if peakpos:
        ax.set_autoscale_on(False)
        ax.plot(x[peakpos],y[peakpos], 'wx', markersize=20)
    if cax:
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_ylabel(zlabel)


def pretty_array(x):
    """ Return a string from list or array for nice print format.
    """
    return('[%s]' % ', '.join('%.2f' % x_i for x_i in x))

def is_number(s):
    """ Check if s is a number or not.
    """
    try:
        float(s)
        return(True)
    except ValueError:
        return(False)
