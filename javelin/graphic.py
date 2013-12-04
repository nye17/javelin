#Last-modified: 04 Dec 2013 16:42:58

import matplotlib.pyplot as plt
from glob import glob

__all__ = ['figure_handler']

def figure_handler(fig=None, figout=None, figext=None, dpi=None, pad_inches=0.1, transparent=True):
    """ Handle the saving, drawing, and formating of figure objects.

    Parameters
    ----------
    fig: Figure object
        target figure.
    figout: str
        Name of the figure.
    figext: str
        format of the output figure, choose among 'pdf', 'eps', 'eye', and 'png', or it could be any combination of those four, like 'pdfeps' or 'epseys', etc. 
    dpi: int
        dpi of output image file, especially useful for 'png' files (not necessary for vector images like 'pdf' and 'eps').
    pad_inches: float
        padding.
    transparent: bool
        Set true if the transparent axes background is needed. 
        
    Returns
    -------
    printed: bool
        Printed or not.
    """
    printed = False
    if figext is None:
        plt.show()
        printed = True
    elif isinstance(fig, plt.Figure):
        if figout is None:
            for i in xrange(100):
                # figout = "fig"+str(i)
                figout = "fig"+str(i).zfill(2)
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
            fig.savefig(figout+".png", transparent=transparent, format="png", dpi=dpi )
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

