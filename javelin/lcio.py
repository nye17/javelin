# Last-modified: 30 Sep 2013 12:25:36 PM

__all__ = ['combinelc', 'readlc', 'writelc', 'readlc_3c', 'file_len']

""" I/O functions for light curve data. 

The functions were originally written in hopes of avoiding numpy dependencies, so
there are no numpy arrays involved, but the speed should suffice.  """

def combinelc(lctxt, lcdata):
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

    Notes
    -----
    The format of ZY light curve data file is explained in the JAVELIN manual.

    """
    ofile = open(lcdata,"wa")
    # count light curve number by file names provided
    num_lightcurves = len(lctxt) 
    ofile.write(str(num_lightcurves)+"\n")
    for i in range(num_lightcurves):
        # count light curve epochs by file lines
        ofile.write(str(file_len(lctxt[i]))+"\n") 
        ofile.write(open(lctxt[i]).read())
    ofile.close()
    return(None)

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
    for ilc in xrange(nlc):
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

