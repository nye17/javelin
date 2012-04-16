#Last-modified: 16 Apr 2012 01:39:23 PM

import matplotlib.pyplot as plt
from glob import glob



def figure_handler(fig=None, figout=None, figext=None, dpi=None, pad_inches=0.1):
    if figext is None:
        plt.show()
        printed = True
    elif isinstance(fig, plt.Figure):
        if figout is None:
            for i in xrange(100):
                figout = "fig"+str(i)
                if (len(glob(figout+".*")) == 0):
                    break
                if (i == 99):
                    print("too many fig* files in current dir!")
                    printed = False
                    return(printed)
        if "pdf" in figext:
            fig.savefig(figout+".pdf", format="pdf", dpi=dpi)
            printed = True
        if "png" in figext:
            fig.savefig(figout+".png", transparent=True, format="png", dpi=dpi )
            printed = True
        if "eps" in figext:
            fig.savefig(figout+".eps", papertype="a4", format="eps",
                              bbox_inches='tight',
                              pad_inches=pad_inches, dpi=dpi)
            printed = True
        if "eye" in figext:
            plt.show()
            printed = True
        if (printed is False):
            print("savefig failed, currently only pdf, png, and eps allowed")
    return(printed)


