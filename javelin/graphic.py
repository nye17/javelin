#Last-modified: 28 Mar 2012 12:11:00 AM

import matplotlib.pyplot as plt
from glob import glob

""" some utility functions for plotting
"""

def figure_handler(fig=None, figout=None, figext=None):
    if figext is None:
        plt.get_current_fig_manager().toolbar.zoom()
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
            fig.savefig(figout+".pdf", format="pdf")
            printed = True
        if "png" in figext:
            fig.savefig(figout+".png", transparent=True, format="pdf")
            printed = True
        if "eps" in figext:
            fig.savefig(figout+".eps", papertype="a4", format="eps",
                              bbox_inches='tight',
                              pad_inches=0.0)
            printed = True
        if (printed is False):
            print("savefig failed, currently only pdf, png, and eps allowed")
    return(printed)
