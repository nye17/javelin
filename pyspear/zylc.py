import numpy as np

#from pyspear.lcio import jdrangelc, jdmedlc
from lcio import jdrangelc, jdmedlc

class zyLC(object):
    """ zyLC object.
    """
    def __init__(self, zylclist):
        if not isinstance(zylclist, list):
            raise RuntimeError("zylclist has to be a list of lists or arrays")
        self.nlc = len(zylclist)

        self.jlist, self.mlist, self.elist = self.sorteddatalist(zylclist)
        self.nptlist = np.array([a.size for a in self.jlist])
        self.npt = sum(self.nptlist)

        self.jarr, self.marr, self.earr, self.iarr = self.combineddataarr()

        self.rj = jdrangelc(zylclist)[-1] - jdrangelc(zylclist)[0]

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
    print(zylc.jlist)
#    print(zylc.jarr)
#    print(zylc.marr)
#    print(zylc.iarr)
