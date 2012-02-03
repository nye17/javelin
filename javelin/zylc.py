import numpy as np

__all__ = ['zyLC', 'get_data']

from lcio import jdrangelc, jdmedlc, readlc, readlc_3c
import matplotlib.pyplot as plt
import matplotlib.cm as cm

""" load light curve files into a zyLC object.
"""

class zyLC(object):
    """ zyLC object.
    """
    def __init__(self, zylclist, names=None, set_subtractmean=True):
        if not isinstance(zylclist, list):
            raise RuntimeError("zylclist has to be a list of lists or arrays")

        self.nlc = len(zylclist)
        if names is None :
            # simply use the light curve ids as their names
            self.names = [str(i) for i in xrange(self.nlc)]
        else :
            if len(names) != self.nlc :
                raise RuntimeError("names should match the dimension of zylclist")
            else :
                self.names = names

        if(self.nlc == 1):
            self.issingle = True
        else:
            self.issingle = False

        self.jlist, self.mlist, self.elist = self.sorteddatalist(zylclist)

        self.cont_mean     = np.mean(self.mlist[0])
        self.cont_mean_err = np.mean(self.elist[0])
        self.cont_std      = np.std(self.mlist[0])
        if self.cont_mean_err != 0.0 :
            self.cont_SN       = self.cont_std/self.cont_mean_err
        else :
            print("Warning: zero mean error in the continuum?")
        self.cont_cad      = jdmedlc(list(self.jlist[0]))

        if set_subtractmean:
            # usually good for the code health, smaller means means less 
            # possibility of large round-off errors in the matrix computations.
            # note tha it all modifies self.mlist.
            self.blist = self.meansubtraction()
        else:
            self.blist = [0.0]*self.nlc

        self.nptlist = np.array([a.size for a in self.jlist])
        self.npt = sum(self.nptlist)

        self.jarr, self.marr, self.earr, self.iarr = self.combineddataarr()

        self.rj = jdrangelc(zylclist)[-1] - jdrangelc(zylclist)[0]

    def plot(self):
        fig  = plt.figure(figsize=(10, 3*self.nlc))
        #axes = []
        height = 0.90/self.nlc
        for i in xrange(self.nlc) :
            ax = fig.add_axes([0.05, 0.1+i*height, 0.9, height])
            #axes.append(ax)
            mfc = cm.jet(1.*(i-1)/self.nlc)
            ax.errorbar(self.jlist[i], self.mlist[i], yerr=self.elist[i], 
                    ecolor='k', marker="o", ms=4, mfc=mfc, mec='k', ls='None',
                    label=self.names[i])
            ax.set_xlim(self.jarr[0], self.jarr[-1])
            if i == 0 :
                ax.set_xlabel("JD")
            else :
                ax.set_xticklabels([])
            ax.legend(loc=1)
        plt.show()
            
            



    def meansubtraction(self):
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
        for lclist in zylclist:
            if (len(lclist) == 3):
                jsubarr, msubarr, esubarr = [np.array(l) for l in lclist]
                # sort the date
                p = jsubarr.argsort()
                jlist.append(jsubarr[p])
                mlist.append(msubarr[p])
                elist.append(esubarr[p])
            else:
                raise RuntimeError("each sub-zylclist has to be a list of lists or arrays")
        return(jlist, mlist, elist)
        
    def combineddataarr(self):
        jarr = np.empty(self.npt)
        marr = np.empty(self.npt)
        earr = np.empty(self.npt)
        iarr = np.empty(self.npt, dtype=int)
        larr = np.zeros((self.nlc, self.npt))
        start = 0
        for i, nptlc in enumerate(self.nptlist):
            jarr[start:start+nptlc] = self.jlist[i]
            marr[start:start+nptlc] = self.mlist[i]
            earr[start:start+nptlc] = self.elist[i]
            # comply with the fortran version, where i starts from 1 rather than 0.
            iarr[start:start+nptlc] = i + 1
            start = start+nptlc
        p = jarr.argsort()
        return(jarr[p], marr[p], earr[p], iarr[p])


def get_data(lcfile, names=None, set_subtractmean=True):
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
    zydata = zyLC(lclist, names=names, set_subtractmean=set_subtractmean)
    return(zydata)


if __name__ == "__main__":    
    zylclist= [[[2.0, 1.0, 5.0, 10.0], [5.0, 5.5, 4.3, 5.6], [0.1, 0.1, 0.1, 0.4]], [[1.5], [5.0], [0.1]], [[8.0, 9.0], [3.0, 1.5], [0.2, 0.1]]]
    zylc = zyLC(zylclist=zylclist)
    print(zylc.cont_cad)
