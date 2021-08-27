"""
Readers for various direct access formatted time series files
"""
from struct import unpack
import os
import numpy as np


def read_ts_names(path):
    """
    Read time series names from key-file associated with the direct access (binary) time series file format '.ts'.

    Parameters
    ----------
    path : str
        Key file path

    Returns
    -------
    list
        Time series names

    Notes
    -----
    Names are stored on ASCII file as one name per line. The file is terminated by END.

    The associated time series are stored on a binary direct access file with suffix '.ts'.
    """
    # read line-separated keys from ascii file
    names = _read_names(path)

    # skip the key referring to the time array (always the first one on the ascii file)
    return names[1:]


def read_tda_names(path):
    """
    Read time series names from key-file associated with the direct access (binary) time series file format '.tda'.

    Parameters
    ----------
    path : str
        Key file path

    Returns
    -------
    list
        Time series names

    Notes
    -----
    The associated time series are stored on a binary direct access file with suffix '.tda'.
    """
    # Read line-separated names from ascii file
    names = _read_names(path)

    # Skip the keys referring to the
    # - info array (always the first one on the ascii file)
    # - time array (always the second one on the ascii file)
    return names[2:]


def read_dis_names(path):
    """
    Read time series names from key-file associated with the direct access (binary) cycle distribution file format '.dis'.

    Parameters
    ----------
    path : str
        Key file path

    Returns
    -------
    list
        Time series names

    Notes
    -----
    The associated cycle distributions are stored on a binary direct access file with suffix '.dis'.
    """
    # read line-separated names from ascii file
    names = _read_names(path)

    return names


def _read_names(path):
    """
    Read line-separated time series names from ascii file

    Parameters
    ----------
    path : str
        Key file path

    Returns
    -------
    list
        Time series names

    Notes
    -----
    Keys are stored on ASCII file as one key per line. The file is terminated by END.
    """
    with open(os.path.join(path, path), 'r') as f:
        names = [l.strip() for l in f if not l.startswith(("**", "'")) and not l.upper().strip() == "END"]

    return names


def read_ts_data(path, ind=None, verbose=False):
    """
    Read time series from binary direct access formatted '.ts' file.

    Parameters
    ----------
    path : str
        File path (relative or absolute)
    ind : list of integers, optional
        Index of the requested records on the file. By default all time series are read.
    verbose : bool, optional
        Increase verbosity

    Returns
    -------
    array
        Time and data

    Examples
    --------
    >>> data = read_ts_data('data.ts')
    >>> t = data[0,:]   # time
    >>> x1 = data[1,:]  # first data series

    Notes
    -----
    Note that `0` is the index of the time array. If ``ind`` is specified, time array is only included if
    `0` is included in the specified indices.
    """
    # the format specifier for decoding binary integers is 'i'
    data = _read_data(path, ifmt='i', ind=ind, verbose=verbose)

    return data


def read_tda_data(path, ind=None, verbose=False):
    """
    Read time series from binary direct access formatted '.tda' file.

    Parameters
    ----------
    path : str
        File path (relative or absolute)
    ind : list of integers, optional
        Index of the requested records on the file. By default all time series are read.
    verbose : bool, optional
        Increase verbosity

    Returns
    -------
    array
        Time and data

    Examples
    --------
    >>> data = read_tda_data('data.tda')
    >>> t = data[0,:]   # time
    >>> x1 = data[1,:]  # first data series

    Notes
    -----
    Note that `0` is the index of the time array. If ``ind`` is specified, time array is only included if
    `0` is included in the specified indices.
    """
    # the format specifier for decoding binary integers is 'i'
    data = _read_data(path, ifmt='f', ind=ind, verbose=verbose)

    return data


def read_dis_data(path, ind=None, verbose=False):
    """
    Read cycle distributions from binary direct access formatted '.dis' file.

    Parameters
    ----------
    path : str
        File path (relative or absolute)
    ind : list of integers, optional
        Index of the requested records on the file. By default all time series are read.
    verbose : bool, optional
        Increase verbosity

    Returns
    -------
    array
        Cycle widths and counts

    Notes
    -----
    Note that `0` is the index of the time array. If ``ind`` is specified, time array is only included if
    `0` is included in the specified indices.
    """
    # the format specifier for decoding binary integers is 'i'
    data = _read_data(path, ifmt='i', ind=ind, verbose=verbose)

    return data


def _read_data(path, ifmt, ind=None, verbose=False):
    """
    Read direct access formatted files with time series or cycle distributions

    Parameters
    ----------
    path : str
        File path (relative or absolute)
    ifmt : str
        Format to decode binary integers
    ind : list of integers, optional
        Index of the requested records on the file. Note that `0` is the index of the time array. All data
        is read by default.
    verbose : bool, optional
        Increase verbosity

    Returns
    -------
    array
        Data

    Notes
    -----
    If ``ind`` is specified, time array is only included if `0` is included in the specified indices.

    If ``ind`` is specified, the response arrays (1-D) are stored in the returned 2-D array in the same order as
    specified. I.e. if ``ind=[0,10,2,3]``, then the response array with index `10` on .ts file is obtained from
    ``arr[1,:]``.

    For reading of .dis files, keep in mind that there is no common time array or similar for each data set. For
    example; if one intends to request data set number three, one would need to request indices 4 and 5 (first set is
    stored on indices 0 and 1, second on 2 and 3, etc.)

    The correct integer format varies with OS and the software exporting the data to file.
        - .ts on Windows  : 'i', 'l'
        - .ts on Linux    : 'i'
        - .tda on Windows : 'f'
        - .tda on Linux   : ?
        - .dis on Windows : 'i'
        - .dis on Linux : ?
    """
    if verbose:
        print('Reading %s ...' % path)

    # indices --> list or None
    if isinstance(ind, int):
        ind = [ind]

    # check extension
    if ifmt not in ("i", "l", "f"):
        raise ValueError(f"Invalid format for decoding binary integers: {ifmt}")

    try:
        with open(path, 'rb') as f:
            # read ts-file information (1st line)
            nbytes = 4
            s = f.read(nbytes)
            ndat = unpack(ifmt, s)[0]  # number of time steps per array

            # the code below works generally, but for some direct access files: nts=nrec-1
            # therefore, `nrec` is based on parsing the file instead
            #
            #   s = f.read(nbytes)
            #   nrec = unpack(ifmt, s)[0]  # number of records(including info record and time record)
            #   nts = nrec-2

            f.seek(0, 2)  # go to end of file
            epos = f.tell()  # get end position (in bytes)
            nrec = epos / (ndat * nbytes)  # no. of "rows"
            nts = nrec - 2  # excl. time array
            # make integers (necessary for ifmt='f')
            ndat = int(ndat)
            nts = int(nts)

            # extract all keys/indices, if <ind> not specified
            if ind is None:
                ind = list(range(nts + 1))  # (including time array)
            else:
                # check number of keys (+1 is due to bug in some direct access files), if specified by user
                assert max(ind) <= nts, "Requested time series no. %d, but there are only %d time series on file" % \
                                        (max(ind), nts)

            # define position
            pos = dict(zip(ind, range(len(ind))))

            # info
            if verbose:
                print('------------------------------------')
                print('nrec (no. of keys on .ts)   : %d' % nts)
                print('ndat (no. of time steps)    : %d' % ndat)
                print('number of keys to read      : %d  (%s)' % (len(ind), 'including time array index 0 is specified'))
                # minus 1 due to time array #len(keys)

            # initiate array
            arr = np.zeros((len(ind), ndat))
            # format string
            fmtstr = 'f' * ndat
            # go to second "row"
            f.seek(nbytes * ndat)
            # read arrays
            j = 0
            for i in range(nts + 1):
                if i in ind:
                    s = f.read(nbytes * ndat)
                    p = pos[i]
                    arr[p, :] = np.array(unpack(fmtstr, s))
                    j += 1
                else:
                    # seek position of next row
                    f.seek(4 * ndat, 1)  # (1 ==> seek relative to current position)

    except OverflowError:
        raise
    except AssertionError:
        raise

    return arr


