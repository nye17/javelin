import numpy as np

from lcio import jdrangelc, jdmedlc

class zyLC(object):
    """ zyLC object.
    """
    def __init__(self, zylclist, set_subtractmean=True):
        if not isinstance(zylclist, list):
            raise RuntimeError("zylclist has to be a list of lists or arrays")
        self.nlc = len(zylclist)

        if(self.nlc == 1):
            self.issingle = True
        else:
            self.issingle = False

        self.jlist, self.mlist, self.elist = self.sorteddatalist(zylclist)

        self.cont_mean     = np.mean(self.mlist[0])
        self.cont_mean_err = np.mean(self.elist[0])
        self.cont_std      = np.std(self.mlist[0])
#        self.cont_avg = np.average(self.mlist[0], 
#                                   weights=1./np.power(self.elist[0], 2))
#        self.cont_std = np.sqrt(np.sum(self.mlist[0] - self.cont_avg)**2)
        self.cont_SN  = self.cont_std/self.cont_mean_err

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

        




            




if __name__ == "__main__":    
    zylclist= [[[2.0, 1.0], [5.0, 5.5], [0.1, 0.1]], [[1.5], [5.0], [0.1]], [[8.0, 9.0], [3.0, 1.5], [0.2, 0.1]]]
    zylc = zyLC(zylclist=zylclist)
    print(zylc.cont_mean_err)
    print(zylc.cont_std)
    print(zylc.cont_SN)
