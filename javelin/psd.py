from __future__ import with_statement
import os
import warnings
import pickle
import math
import numpy as np

def getMatplotlibVersion():
    """
    Get matplotlib version information.

    :returns: Matplotlib version as a list of three integers or ``None`` if
        matplotlib import fails.
    """
    try:
        import matplotlib
        version = matplotlib.__version__.replace('svn', '')
        version = matplotlib.__version__.replace('svn', '')
        version = map(int, version.split(".")[:2])
    except ImportError:
        version = None
    return version

MATPLOTLIB_VERSION = getMatplotlibVersion()

if MATPLOTLIB_VERSION == None:
    # if matplotlib is not present be silent about it and only raise the
    # ImportError if matplotlib actually is used (currently in psd() and
    # PPSD())
    msg_matplotlib_ImportError = "Failed to import matplotlib. While this " \
            "is no dependency of obspy.signal it is however necessary for a " \
            "few routines. Please install matplotlib in order to be able " \
            "to use e.g. psd() or PPSD()."
    # set up two dummy functions. this makes it possible to make the docstring
    # of psd() look like it should with two functions as default values for
    # kwargs although matplotlib might not be present and the routines
    # therefore not usable

    def detrend_none():
        pass

    def window_hanning():
        pass

else:
    # Import matplotlib routines. These are no official dependency of
    # obspy.signal so an import error should really only be raised if any
    # routine is used which relies on matplotlib (at the moment: psd, PPSD).
    from matplotlib import mlab
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.mlab import detrend_none, window_hanning


# do not change these variables, otherwise results may differ from PQLX!
PPSD_LENGTH = 3600  # psds are calculated on 1h long segments
PPSD_STRIDE = 1800  # psds are calculated overlapping, moving 0.5h ahead


def psd(x, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
        noverlap=0):
    """
    Wrapper for :func:`matplotlib.mlab.psd`.

    Always returns a onesided psd (positive frequencies only), corrects for
    this fact by scaling with a factor of 2. Also, always normalizes to dB/Hz
    by dividing with sampling rate.

    This wrapper is intended to intercept changes in
    :func:`matplotlib.mlab.psd` default behavior which changes with
    matplotlib version 0.98.4:

    * http://matplotlib.sourceforge.net/users/whats_new.html\
#psd-amplitude-scaling
    * http://matplotlib.sourceforge.net/_static/CHANGELOG
      (entries on 2009-05-18 and 2008-11-11)
    * http://matplotlib.svn.sourceforge.net/viewvc/matplotlib\
?view=revision&revision=6518
    * http://matplotlib.sourceforge.net/api/api_changes.html#changes-for-0-98-x

    .. note::
        For details on all arguments see :func:`matplotlib.mlab.psd`.

    .. note::
        When using `window=welch_taper` (:func:`obspy.signal.psd.welch_taper`)
        and `detrend=detrend_linear` (:func:`matplotlib.mlab.detrend_linear`)
        the psd function delivers practically the same results as PITSA.
        Only DC and the first 3-4 lowest non-DC frequencies deviate very
        slightly. In contrast to PITSA, this routine also returns the psd value
        at the Nyquist frequency and therefore is one frequency sample longer.
    """
    # check if matplotlib is available, no official dependency for obspy.signal
    if MATPLOTLIB_VERSION is None:
        raise ImportError(msg_matplotlib_ImportError)

    # check matplotlib version
    elif MATPLOTLIB_VERSION >= [0, 99]:
        new_matplotlib = True
    else:
        new_matplotlib = False
    # build up kwargs that do not change with version 0.98.4
    kwargs = {}
    kwargs['NFFT'] = NFFT
    kwargs['Fs'] = Fs
    kwargs['detrend'] = detrend
    kwargs['window'] = window
    kwargs['noverlap'] = noverlap
    # add additional kwargs to control behavior for matplotlib versions higher
    # than 0.98.4. These settings make sure that the scaling is already done
    # during the following psd call for newer matplotlib versions.
    if new_matplotlib:
        kwargs['pad_to'] = None
        kwargs['sides'] = 'onesided'
        kwargs['scale_by_freq'] = True
    # do the actual call to mlab.psd
    Pxx, freqs = mlab.psd(x, **kwargs)
    # do scaling manually for old matplotlib versions
    if not new_matplotlib:
        Pxx = Pxx / Fs
        Pxx[1:-1] = Pxx[1:-1] * 2.0
    return Pxx, freqs
