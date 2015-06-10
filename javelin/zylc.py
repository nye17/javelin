# Last-modified: 06 Dec 2013 01:58:44

__all__ = ['LightCurve', 'get_data']

from lcio import readlc, readlc_3c, writelc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphic import figure_handler
import numpy as np
from numpy.random import normal, multivariate_normal
from copy import copy, deepcopy

""" Load light curve files into a LightCurve object.
"""

class LightCurve(object):
    def __init__(self, zylclist, names=None, set_subtractmean=True, qlist=None):
        """ LightCurve object for encapsulating light curve data.

        Parameters
        ----------
        zylclist: list of lists/ndarrays
            List of light curves.

        names: list of strings, optional
            Names of each individual light curve (default: None).

        set_subtractmean: bool, optional
            True if light curve means are subtracted (default: True).

        qlist: list of q values, optional
            Best-fit q values from javelin fitting, for accounting for the difference between sample means and 
            the truth means of light curves (default: None).

        """
        if not isinstance(zylclist, list):
            raise RuntimeError("zylclist has to be a list of lists or arrays")
        else :
            self.zylclist = []
            for i in xrange(len(zylclist)) :
                if isinstance(zylclist[i], np.ndarray) :
                    if zylclist[i].shape[-1] == 3 :
                        # if each element is an ndarray, convert to a list of 1d arrays
                        self.zylclist.append([zylclist[i][:,0], zylclist[i][:,1], zylclist[i][:,2]])
                    else :
                        raise RuntimeError("each single light curve array should have shape (#, 3)")
                elif isinstance(zylclist[i], list) :
                    if len(zylclist[i]) == 3 :
                        self.zylclist.append([zylclist[i][0], zylclist[i][1], zylclist[i][2]])
                    else :
                        raise RuntimeError("each single light curve list should have 3 ndarrays")
        # number of light curves
        self.nlc = len(self.zylclist)
        if names is None :
            # simply use the sequences as their names (start from 0)
            self.names = [str(i) for i in xrange(self.nlc)]
        else :
            if len(names) != self.nlc :
                raise RuntimeError("names should match the dimension of zylclist")
            else :
                self.names = names

        # issingle makes a difference in what you can do with the light curves
        if(self.nlc == 1):
            self.issingle = True
        else:
            self.issingle = False

        # jlist/mlist/elist/ilist: list of j, m, e, i of each individual light curve 
        self.jlist, self.mlist, self.elist, self.ilist = self.sorteddatalist(self.zylclist)

        # continuum properties, useful in determining continuum variability
        self.cont_mean     = np.mean(self.mlist[0])
        self.cont_mean_err = np.mean(self.elist[0])
        self.cont_std      = np.std(self.mlist[0])
        self.cont_cad_arr  = self.jlist[0][1 :] - self.jlist[0][:-1]  
        self.cont_cad      = np.median(self.cont_cad_arr)
        self.cont_cad_min  = np.min(self.cont_cad_arr)
        self.cont_cad_max  = np.max(self.cont_cad_arr)
        if self.cont_mean_err != 0.0 :
            # a rough estimate of the continuum variability signal-to-noise
            # ratio
            self.cont_SN = self.cont_std/self.cont_mean_err
        else :
            self.cont_SN = np.inf

        # subtract the mean to get *blist*
        # usually good for the code health, smaller means means less 
        # possibility of large round-off errors in the matrix computations.
        # note tha it also modifies self.mlist.
        if set_subtractmean:
            self.blist = self.meansubtraction()
        else:
            self.blist = [0.0]*self.nlc

        # number statistics: nptlist, npt
        self.nptlist = np.array([a.size for a in self.jlist])
        self.npt = sum(self.nptlist)
        
        # combine all information into one vector, those are the primariy
        # vectors we are gonna use in spear covariance.
        self.jarr, self.marr, self.earr, self.iarr = self.combineddataarr()
        # variance array
        self.varr = self.earr*self.earr

        # construct the linear response matrix
        self.larr   = np.zeros((self.npt, self.nlc))
        for i in xrange(self.npt):
            lcid = self.iarr[i] - 1
            self.larr[i, lcid] = 1.0
        self.larrTr = self.larr.T

        # baseline of all the light curves
        self.jstart = self.jarr[0]
        self.jend   = self.jarr[-1]
        self.rj = self.jend - self.jstart

        # q values for the *true* means of light curves
        self.qlist = self.nlc*[0.0]
        if qlist is None :
            pass
        else :
            if len(qlist) != self.nlc :
                raise RuntimeError("qlist should match the dimension of zylclist")
            else :
                self.update_qlist(qlist)

    def __add__(self, other) :
        _zylclist = self.zylclist + other.zylclist
        _names    = self.names + other.names
        return(LightCurve(_zylclist, names=_names))

    def shed_continuum(self) :
        _zylclist = [self.zylclist[0],]
        _names    = [self.names[0],]
        return(LightCurve(_zylclist, names=_names))

    def split(self) :
        """ split into individual LightCurves objects whenever the parent has multiple lightcurves.
        """
        eggs = []
        for i in xrange(self.nlc) :
            _zylclist = [self.zylclist[i],]
            _names    = [self.names[i],]
            eggs.append(LightCurve(_zylclist, names=_names))
        return(eggs)

    def spawn(self, errcov=0.0, names=None) : 
        """ generate one LightCurve for which the lightcurve values are the sum of the original ones and gaussian variates from gaussian errors.
        """
        # _zylclist = list(self.zylclist) # copy the original list
        _zylclist = deepcopy(self.zylclist) # copy the original list
        for i in xrange(self.nlc) :
            e = np.atleast_1d(_zylclist[i][2])
            nwant = e.size
            ediag = np.diag(e*e)
            if errcov == 0.0 :
                ecovmat = ediag
            else :
                temp1 = np.repeat(e, nwant).reshape(nwant,nwant)
                temp2 = (temp1*temp1.T - ediag)*errcov
                ecovmat = ediag + temp2
            et = multivariate_normal(np.zeros_like(e), ecovmat)
            _zylclist[i][1] = _zylclist[i][1] + et
        if names is None :
            names = ["-".join([r, "mock"]) for r in self.names]
        return(LightCurve(_zylclist, names=names))

    def shift_time(self, timeoffset) :
        """ shift the time axies by `timeoffset`
        """
        # fix jarr
        self.jarr = self.jarr + timeoffset
        self.jstart = self.jarr[0]
        self.jend   = self.jarr[-1]
        # fix jlist and zylclist
        for i in xrange(self.nlc) :
            # fix jlist
            self.jlist[i] = self.jlist[i] + timeoffset
            # fix the original zylclist
            self.zylclist[i][0] = np.atleast_1d(zylclist[i][0]) + timeoffset

    def plot(self, set_pred=False, obs=None, marker="o", ms=4, ls='None', lw=1, figout=None, figext=None) :
        """ Plot light curves.

        Parameters
        ----------
        set_pred: bool, optional
            True if the current light curve data are simulated or predicted from
            javelin, rather than real data (default: False).

        obs: bool, optional
            The observed light curve data to be overplotted on the current light
            curves, usually used when set_pred is True (default: None).

        marker: str, optional
            Marker symbol (default: 'o').

        ms : float, optional
            Marker size (default: 4).

        ls : str, optional
            Line style (default: 'None').

        lw: float, optional
            Line width (default: 1).

        figout: str, optional
            Output figure name (default: None).

        figext: str, optional
            Output figure extension, ``png``, ``eps``, or ``pdf``. Set to None
            for drawing without saving to files (default: None)


        """
        fig  = plt.figure(figsize=(8, 2*self.nlc))
        height = 0.85/self.nlc
        for i in xrange(self.nlc) :
            ax = fig.add_axes([0.10, 0.1+i*height, 0.85, height])
            mfc = cm.jet(i/(self.nlc-1.) if self.nlc > 1 else 0)
            if set_pred :
                ax.plot(self.jlist[i], self.mlist[i]+self.blist[i],
                    color=mfc, ls="-", lw=2,
                    label=self.names[i])

                ax.fill_between(self.jlist[i],
                    y1=self.mlist[i]+self.blist[i]+self.elist[i], 
                    y2=self.mlist[i]+self.blist[i]-self.elist[i], 
                    color=mfc, alpha=0.5,
                    label=self.names[i])
                if obs is not None :
                    ax.errorbar(obs.jlist[i], obs.mlist[i]+obs.blist[i], 
                            yerr=obs.elist[i], 
                            ecolor='k', marker=marker, ms=ms, mfc=mfc, mec='k', ls=ls, lw=lw,
                            label=" ".join([self.names[i], "observed"]))
            else :
                if np.sum(self.elist[i]) == 0.0 :
                    # no error, pure signal.
                    ax.plot(self.jlist[i], self.mlist[i]+self.blist[i], 
                        marker=marker, ms=ms, mfc=mfc, mec='k', ls=ls, lw=lw,
                        label=self.names[i], color=mfc)
                else :
                    ax.errorbar(self.jlist[i], self.mlist[i]+self.blist[i], 
                        yerr=self.elist[i], 
                        ecolor='k', marker=marker, ms=ms, mfc=mfc, mec='k', ls=ls, lw=lw,
                        label=self.names[i])

            ax.set_xlim(self.jstart, self.jend)
            ax.set_ylim(np.min(self.mlist[i])+self.blist[i]-np.min(self.elist[i]),
                        np.max(self.mlist[i])+self.blist[i]+np.max(self.elist[i]))
            if i == 0 :
                ax.set_xlabel(r"$t$")
            else :
                ax.set_xticklabels([])
            ax.set_ylabel(r"$f$")
            leg = ax.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(0.5)
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def plotdt(self, set_logdt=False, figout=None, figext=None, **histparams) :
        """ Plot the time interval distribution.

        set_logdt: bool, optional
            True if Delta t is in log (default: False)

        figout: str, optional
            Output figure name (default: None).

        figext: str, optional
            Output figure extension, ``png``, ``eps``, or ``pdf``. Set to None
            for drawing without saving to files (default: None)

        histparams: kargs, optional
            Parameters for ax.hist.

        """
        _np  = self.nptlist[0]
        _ndt = _np*(_np-1)/2
        dtarr = np.zeros(_ndt)
        _k = 0
        for i in xrange(_np-1) :
            for j in xrange(i+1, _np) :
                dtarr[_k] = self.jlist[0][j] - self.jlist[0][i]
                _k += 1
        if set_logdt :
            dtarr = np.log10(dtarr)
        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.hist(dtarr, **histparams)
        if set_logdt :
            ax.set_xlabel(r"$\log\;\Delta t$")
        else :
            ax.set_xlabel(r"$\Delta t$")
        return(figure_handler(fig=fig, figout=figout, figext=figext))

    def save(self, fname, set_overwrite=True):
        """ Save current LightCurve into zylc file format.

        Parameters
        ----------
        fname : str
            Output file name.

        set_overwrite: bool, optional
            True to overwrite existing files (default: True).

        """
        try :
            f=open(fname, "r")
            if not set_overwrite :
                raise RuntimeError("%s exists, exit"%fname)
            else :
                print("save light curves to %s"%fname)
                writelc(self.zylclist, fname)
        except IOError :
            print("save light curves to %s"%fname)
            writelc(self.zylclist, fname)

    def save_continuum(self, fname, set_overwrite=True):
        """ Save the continuum part of LightCurve into zylc file format.

        Parameters
        ----------
        fname : str
            Output file name.

        set_overwrite: bool, optional
            True to overwrite existing files (default: True).

        """
        try :
            f=open(fname, "r")
            if not set_overwrite :
                raise RuntimeError("%s exists, exit"%fname)
            else :
                print("save continuum light curves to %s"%fname)
                writelc([self.zylclist[0]], fname)
        except IOError :
            print("save continuum light curves to %s"%fname)
            writelc([self.zylclist[0]], fname)

    def save_lcarr(self, fname, set_overwrite=True, set_addmean=True, set_saveid=False) :
        """ Save the data array into a 3-column file.
         
        Parameters
        ----------
        fname : str
            Output file name.

        set_overwrite: bool, optional
            True to overwrite existing files (default: True).

        set_saveid: bool, optional
            True to save id array (default: False).

        set_addmean: bool, optional
            True to save original light curve values without no mean subtraction (default: True).
        """
        try :
            f=open(fname, "r")
            if not set_overwrite :
                raise RuntimeError("%s exists, exit"%fname)
        except IOError :
            print("%s does not exist")
        print("save light curve data array to %s"%fname)

        # TODO
        _marr = np.empty(self.npt)
        if set_addmean :
            for i in xrange(self.nlc) :
                sel = (self.iarr == i+1)
                _marr[sel] = self.marr[sel] + self.blist[i]
        else :
            _marr = self.marr
        if set_saveid :
            np.savetxt(fname, np.vstack((self.jarr, _marr, self.earr, self.iarr)).T)
        else :
            np.savetxt(fname, np.vstack((self.jarr, _marr, self.earr)).T)

    def update_qlist(self, qlist_new):
        """ Update blist and mlist of the LightCurve object according to the 
        newly acquired qlist values. 

        Parameters
        ----------
        qlist_new: list
            Best-fit light curve mean modulation factors.

        """
        for i in xrange(self.nlc):
            # recover original data when qlist=0
            self.blist[i] -= self.qlist[i]
            self.mlist[i] += self.qlist[i]
            # add      q to   blist
            # subtract q from mlist
            self.blist[i] += qlist_new[i]
            self.mlist[i] -= qlist_new[i]
        # redo combineddataarr
        self.jarr, self.marr, self.earr, self.iarr = self.combineddataarr()
        # variance array
        self.varr = self.earr*self.earr
        # update qlist
        self.qlist = qlist_new

    def meansubtraction(self):
        """ Subtract the mean.

        Returns
        -------
        blist: list
            list of light curve means.

        """
        blist = []
        for i in xrange(self.nlc):
            bar = np.mean(self.mlist[i])
            blist.append(bar)
            self.mlist[i] = self.mlist[i] - bar
        return(blist)

    def sorteddatalist(self, zylclist):
        """ Sort the input lists by time epochs.

        Parameters
        ----------
        zylclist: list of lists/ndarrays
            List of light curves.

        Returns
        -------
        jlist:
            List of time ndarrays.

        mlist:
            List of flux/mag ndarrays.

        elist:
            List of error ndarrays.

        ilist:
            List of index ndarrays, starting at 1 rather than 0.

        """
        jlist = []
        mlist = []
        elist = []
        ilist = []
        for ilc, lclist in enumerate(zylclist):
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                nptlc = len(jsubarr)
                # sort the date
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
                ilist.append(np.zeros(nptlc, dtype="int")+ilc+1)
            else:
                raise RuntimeError("each sub-zylclist has to be a list of lists or arrays")
        return(jlist, mlist, elist, ilist)
        
    def combineddataarr(self):
        """ Combine lists into ndarrays.

        Returns
        -------
        jarr:
            Sorted time ndarray.

        marr:
            Sorted tFlux/mag ndarray.

        earr:
            Sorted tError ndarray.

        iarr:
            Sorted tindex ndarray.

        """
        jarr = np.empty(self.npt)
        marr = np.empty(self.npt)
        earr = np.empty(self.npt)
        iarr = np.empty(self.npt, dtype="int")
        larr = np.zeros((self.nlc, self.npt))
        start = 0
        for i, nptlc in enumerate(self.nptlist):
            jarr[start:start+nptlc] = self.jlist[i]
            marr[start:start+nptlc] = self.mlist[i]
            earr[start:start+nptlc] = self.elist[i]
            # comply with the fortran version, where i starts from 1 rather than 0.
            #iarr[start:start+nptlc] = i + 1
            iarr[start:start+nptlc] = self.ilist[i]
            start = start+nptlc
        p = jarr.argsort()
        return(jarr[p], marr[p], earr[p], iarr[p])

def get_data(lcfile, names=None, set_subtractmean=True, timeoffset=0.0):
    """ Read light curve file(s) into a LightCurve object.

    Parameters
    ----------
    lcfile: string or list of strings
        Input files, could be a single or multiple 3-column light curve files, or
        a single zylc format light curve file.

    names: list of strings
        Names of each files in *lcfile* (default: None).

    set_subtractmean: bool
        Subtract mean in LightCurve if True (default: True).

    timeoffset: float
        The offset added to the time array in `lcfile`, so that t_final = t_orig + timeoffset

    Returns
    -------
    zydata: LightCurve object
        Combined data in a LightCurve object

    """
    if isinstance(lcfile, basestring):
        nlc = 1
        # lcfile should be a single file
        try :
            # either a 3-column single lc file
            lclist = readlc_3c(lcfile)
        except :
            # or a zylc file
            lclist = readlc(lcfile)
    else :
        # lcfile should be a list or tuple of 3-column files
        try :
            nlc = len(lcfile)
        except :
            raise RuntimeError("input is neither a list/tuple nor a string?")
        lclist = []
        for lcf in lcfile :
            lc = readlc_3c(lcf)
            lclist.append(lc[0])
    for ilc in xrange(len(lclist)) :
        lclist[ilc][0] = np.atleast_1d(lclist[ilc][0]) + timeoffset
    zydata = LightCurve(lclist, names=names, set_subtractmean=set_subtractmean)
    return(zydata)


if __name__ == "__main__":    
    zylclist= [
               [
                [2.0, 1.0, 5.0, 10.0], 
                [5.0, 5.5, 4.3,  5.6], 
                [0.1, 0.1, 0.1,  0.4]
               ], 
               [
                [1.5], 
                [5.0], 
                [0.1]
               ], 
               [
                [8.0, 9.0], 
                [3.0, 1.5], 
                [0.2, 0.1]
               ]
              ]
    zylc = LightCurve(zylclist=zylclist)
    print(zylc.cont_cad)
