import numpy as np

__all__ = ['LightCurve', 'get_data']

from lcio import readlc, readlc_3c, writelc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" load light curve files into a LightCurve object.
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
            Best-fit q values from javelin fitting, used to obtain the truth
            mean of light curves (default: None).

        """
        if not isinstance(zylclist, list):
            raise RuntimeError("zylclist has to be a list of lists or arrays")
        else :
            self.zylclist = zylclist
        # number of light curves
        self.nlc = len(zylclist)
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
        self.jlist, self.mlist, self.elist, self.ilist = self.sorteddatalist(zylclist)

        # continuum properties, useful in determining continuum variability
        self.cont_mean     = np.mean(self.mlist[0])
        self.cont_mean_err = np.mean(self.elist[0])
        self.cont_std      = np.std(self.mlist[0])
        self.cont_cad_arr  = self.jlist[0][1:] - self.jlist[0][:-1]  
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


    def plot(self, set_pred=False, obs=None):
        fig  = plt.figure(figsize=(10, 3*self.nlc))
        #axes = []
        height = 0.90/self.nlc
        for i in xrange(self.nlc) :
            ax = fig.add_axes([0.05, 0.1+i*height, 0.9, height])
            mfc = cm.jet(1.*(i-1)/self.nlc)
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
                            ecolor='k', marker="o", ms=4, mfc=mfc, mec='k', ls='None',
                            label=" ".join([self.names[i], "observed"]))
            else :
                ax.errorbar(self.jlist[i], self.mlist[i]+self.blist[i], 
                    yerr=self.elist[i], 
                    ecolor='k', marker="o", ms=4, mfc=mfc, mec='k', ls='None',
                    label=self.names[i])

            ax.set_xlim(self.jstart, self.jend)
            ax.set_ylim(np.min(self.mlist[i])+self.blist[i]-np.min(self.elist[i]),
                        np.max(self.mlist[i])+self.blist[i]+np.max(self.elist[i]))
            if i == 0 :
                ax.set_xlabel("JD")
            else :
                ax.set_xticklabels([])
            ax.legend(loc=1)
        plt.show()
#        plt.draw()

    def save(self, fname, set_overwrite=True):
        """ save zydata into zylc file format.
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
        """ save the continuum part of zydata into zylc file format.
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

    def update_qlist(self, qlist_new):
        """ update blist and mlist of the LightCurve object according to the 
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


def get_data(lcfile, names=None, set_subtractmean=True):
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
