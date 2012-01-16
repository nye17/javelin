# Last-modified: 16 Jan 2012 03:53:29 PM

""" 
Functions related to the processing of quasar light curves. Numpy is not used
here so that it could be called by basic python outside of JAVELIN.

"""
__all__ = ['combinelc', 'readlc', 'readlc_3c', 'writelc', 'plotlc',
'plotlc_single', 'jdrangelc', 'jdmedlc', 'file_len']


try:
    import matplotlib.pyplot as plt
except ImportError:
    print "Error import matplotlib.pyplot, "
    print "graphic modules will not work." 


def combinelc(lctxt,lcdata):
    """ Write a list of 3-column based light curve files into the ZY light curve data file.

    Parameters
    ----------
    lctxt: list
        list of names of the input files.
    
    lcdata: string
        name of the new ZY light curve file.

    Examples
    --------
    Convert a continuum plus a line light curves into a ZY light
    curve file.

    >>> combinelc(["mrk50_con.dat", "mrk50_hb.dat"], "lc_mrk50.dat")

    """
    ofile = open(lcdata,"wa")
    num_lightcurves = len(lctxt)
    ofile.write(str(num_lightcurves)+"\n")
    for i in range(num_lightcurves):
        ofile.write(str(file_len(lctxt[i]))+"\n")
        ofile.write(open(lctxt[i]).read())
    ofile.close()
    # success
    return(0)


def readlc(lcdata):
    """ Read the ZY light curve data file into a list of lists.

    Parameters
    ----------
    lcdata: string
        ZY light curve file or opened file handle.

    Returns
    -------
    lightcurvearray: list
        List of 3-column lists.

    Examples
    --------
    Read a ZY lc file with 2 light curve and 3 epochs.

    >>> import StringIO
    >>> MESSAGE = '2\\n 2\\n 1. 5.0 0.1\\n 2. 5.5 0.1\\n 1\\n 1.5 15.0 0.1'
    >>> lcdata = StringIO.StringIO(MESSAGE)
    >>> readlc(lcdata)
    [[[1.0, 2.0], [5.0, 5.5], [0.1, 0.1]], [[1.5], [15.0], [0.1]]]

    Read a ZY lc file with 1 light curve and 2 epochs.

    >>> MESSAGE = '1\\n 2\\n 1. 5.0 0.1\\n 2. 5.5 0.1'
    >>> lcdata = StringIO.StringIO(MESSAGE)
    >>> readlc(lcdata)
    [[[1.0, 2.0], [5.0, 5.5], [0.20, 0.30]]]

    .. warning ::
        
        The function reads a single light curves as list of a 3-column 
        list as well, i.e., the light curve can only be extracted by
        readlc(lcdata)[0] rather than readlc(lcdata).

    """
    try:
        f = open(lcdata, "r")
    except:
        # assume it is an opened file handle object.
        f = lcdata
    num_lightcurves = int(f.readline().strip())
    lightcurvelist = []
    for i in range(num_lightcurves):
        num_datapts = int(f.readline().strip())
        lightcurvelist.append([f.readline().strip().split() for j in 
                               range(num_datapts)])
    lightcurvearray = []
    for lightcurve in lightcurvelist:
        jdlist    = []
        fluxlist  = []
        errorlist = []
        for datapoint in lightcurve:
            datapoint=list(map(float,datapoint))
            jdlist.append(datapoint[0])
            fluxlist.append(datapoint[1])
            errorlist.append(datapoint[2])
        lightcurvearray.append([jdlist,fluxlist,errorlist])
    f.close()
    return(lightcurvearray)


def readlc_3c(lcdata):
    """ Read the common 3-column light curve data file into a list of one list.

    Parameters
    ----------
    lcdata: string
        ZY light curve file or opened file handle.

    Returns
    -------
    lightcurvearray: list
        List of a 3-column list (redundant but simply try to comply with the norm).
    """
    try:
        f = open(lcdata, "r")
    except:
        f = lcdata
    num_lightcurves = 1
    lightcurvelist = []
    lightcurvelist.append([r.strip().split() for r in f.readlines()])
    lightcurvearray = []
    for lightcurve in lightcurvelist:
        jdlist    = []
        fluxlist  = []
        errorlist = []
        for datapoint in lightcurve:
            datapoint=list(map(float,datapoint))
            jdlist.append(datapoint[0])
            fluxlist.append(datapoint[1])
            errorlist.append(datapoint[2])
        lightcurvearray.append([jdlist,fluxlist,errorlist])
    f.close()
    return(lightcurvearray)


def writelc(lightcurvearray, lcdata, fmt="10.5f"):
    """ Write the list of lists into a ZY light curve data file.

    Parameters
    ----------
    lightcurvearray: list
        List of 3-column lists.

    lcdata: string
        ZY light curve file or opened file handle.

    fmt: string
        Format of output floating numbers

    Examples
    --------
    Write a ZY lc file with 2 light curve and 3 epochs.

    >>> import StringIO
    >>> MESSAGE = ''
    >>> lcdata = StringIO.StringIO(MESSAGE)
    >>> lightcurve = [[[1.0, 2.0], [5.0, 5.5], [0.1, 0.1]], [[1.5], [15.0], [0.1]]]
    >>> writelc(lightcurve, lcdata, fmt="5.2f")

    """
    try:
        f = open(lcdata, "w")
    except:
        # assume it is an opened file handle object.
        f = lcdata
    nlc = len(lightcurvearray)
    f.write(str(nlc)+"\n")
    for ilc in range(nlc):
        lightcurve = lightcurvearray[ilc]
        assert len(lightcurve) == 3, "Input list not consist of 3-column sublists"
        jd, pt, er = lightcurve
        assert len(jd) == len(pt) == len(er), "Input list shape mismatch"
        npt = len(jd)
        f.write(str(npt)+"\n")
        f.write("\n".join([" ".join([format(jd[i], fmt), format(pt[i], fmt),
            format(er[i], fmt)]) for i in range(npt)]))
        f.write("\n")
    f.close()


def plotlc(lcdata, lcpred=None, epsout=None, pdfout=None, ylabel=None):
    """ Plot the ZY light curve data file.

    Parameters
    ----------
    lcdata: string
        ZY light curve file or opened file handle. 
    lcpred: string or list of strings, optional
        ZY light curve file or opened file handle for predicted light
        curves (default is None).
    epsout: string, optinal
        Output EPS file name, onscreen if None (default is None).
    pdfout: string, optinal
        Output PDF file name, onscreen if None (default is None).
    ylabel: string, optinal
        Label for y-axis, default is "Flux" if set to None.

    """
    fig = plt.figure()
    fig.clf()

    lightcurves = readlc(lcdata)
    jdrange     = jdrangelc(lightcurves)
    n_lc        = len(lightcurves)

    axes = [fig.add_subplot(n_lc, 1, i+1) for i in range(n_lc)]

    for lc, ax in zip(lightcurves, axes):
        plotlc_single(lc, ax, jdrange=jdrange, ylabel=ylabel)

    if not lcpred is None:
        line_props = ["g-", "b--", "m:"]
        # a trick to get input from both lists or sing string.
        ipred = 0
        for thislcpred in lcpred if not isinstance(lcpred, basestring) else [lcpred]:
            line_prop = line_props[ipred%3]
            try:
                fakecurves = readlc(thislcpred)
            except:
                fakecurves = readlc_3c(thislcpred)
            n_fc        = len(fakecurves)
            if (n_lc != n_fc):
                raise RuntimeError("unequal shape of lcdata and [one of] lcpred")
            for fc, ax in zip(fakecurves, axes):
                plotlc_single(fc, ax, set_snake=True, set_overplot=True, line_prop=line_prop)
            ipred += 1

    if epsout:
        fig.savefig(epsout, format="eps")
    if pdfout:
        fig.savefig(pdfout, format="pdf")
    if (epsout is None) and (pdfout is None):
        plt.show()


def plotlc_single(lc, ax, jdrange=None, yrange=None, ylabel=None, 
                          set_snake=False, label=None,
                          set_overplot=False, point_color="r", line_prop="g-"):
    """ Plot a single light curve in a single axis frame.

    Parameters
    ----------
    lc: list of 3 lists
        Format is [[jd], [flux or mag], [error]]
    ax: axis object
        Axis assigned for plot
    jdrange: list or tuple
        X-axis limits as in (xmin, xmax), default is None.
    yrange: list or tuple
        Y-axis limits as in (ymin, ymax), default is None.
    ylabel: string, optinal
        Label for y-axis, default is "Flux" if set to None.
    set_snake: boolean, optinal
        True if plot the errors as error snakes (default is False).
    label: string, optinal
        Label for the plotted light curve.
    set_overplot: boolean, optinal
        True if current plot is an overplot onto an existing light
        curve plot, usually set for overplotting predicted light
        curves against data light curves (default is False).
    point_color: string, optinal
        Color of the points (default: "r").
    line_prop: string, optinal
        Properties of the overplotted line (default: "g-").

    """
    x   = lc[0]
    y   = lc[1]
    yerr= lc[2]
    yupp = [(p + q) for p, q in zip(y, yerr)]
    ylow = [(p - q) for p, q in zip(y, yerr)]

    if set_overplot:
        ax.set_autoscale_on(False)

    if set_snake:
        ax.fill_between(x, ylow, yupp, facecolor="gray", alpha=0.5)
        ax.plot(x, y,     line_prop, lw=2, alpha=0.5, label=label)
    else:
        ax.errorbar(x, y, yerr=yerr, ecolor='black', fmt='o',
                    mfc=point_color,marker='o', ms=5, alpha=0.9, label=label)

    if not set_overplot:
        ax.set_xlabel('JD [days]')
        if not ylabel is None:
            ax.set_ylabel(ylabel)
        if not jdrange is None:
            ax.set_xlim(jdrange)
        if not yrange is None:
            ax.set_ylim(yrange)


def jdrangelc(lightcurves):                                               
    """ Find the JD range of the light curves.                            

    Parameters
    ----------
    lightcurves: list
        List of 3-column lists.

    Returns
    -------
    jdminmax: tuple
        (jd_min,jd_max)

    Examples
    --------
    Find the JD range of two light curves in ZY format.

    >>> lcs = [[[1.0, 2.0], [5.0, 5.5], [0.20, 0.30]], [[1.5], [15.0], [0.1]]]
    >>> jdrangelc(lcs)
    (1.0, 2.0)

    """                                                                 
    jdtotal   = []                                                      
    for singlelc in lightcurves:                                         
        jdtotal  +=(singlelc[0])                                        
    xmin = min(jdtotal)                                                 
    xmax = max(jdtotal)                                                 
    return((xmin,xmax))


def jdmedlc(jddates):                                               
    """ Find the median sampling interval of the light curve

    Parameters
    ----------
    jddates: list
        Epoches of Light curve data points, can be unsorted.

    Returns
    -------
    tmed: float
        Median sampling interval

    Examples
    --------
    Find the median sampling interval of a continuum light curve.

    >>> dates = [2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 1.0]
    >>> jdmedlc(dates)
    >>> 4.0

    """
    # sort the dates
    jddates.sort()
    n = len(jddates)
    delts = []
    for i in xrange(n-1):
        delts.append(jddates[i+1] - jddates[i])
    # sort the intervals
    delts.sort()
    nmed = int(n*0.5-0.5)
    tmed = delts[nmed]
    return(tmed)


def file_len(fname):
    ''' Calculate the number of lines in a file.

    Parameters
    ----------
    fname: string
        Name of input file.

    Returns
    -------
    l: scalar
        Number of lines.
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return(i + 1)

