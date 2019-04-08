"""
Functions for reading various time series file formats.
"""
from struct import unpack
import os
import numpy as np


def read_ts_keys(keyfile, path='', filetype='ts'):
    """
    Read key-files for ts-files on format 'ts' and 'tda'.

    Parameters
    ----------
    keyfile : str
        Name of keyfile.
    path : str, optional
        Path to file
    filetype : {'ts', 'tda', 'dis'}, optional
        Type of corresponding time series file
            ts - Direct access time series file with time array and without info array
            tda - Direct access time series file with time array and info array
            dis - Direct access cycle distribution file without time array and info array

    Returns
    -------
    list
        List of key entries on key-file.

    """
    if filetype not in ("ts", "tda", "dis"):
        raise ValueError("Unknown file format specified: %s" % filetype)

    keys = []
    with open(os.path.join(path, keyfile), 'r') as f:
        for line in f:
            if not line.startswith(("**", "'")) and not line.upper().strip() == "END":
                keys.append(line.strip())

    if filetype == 'tda':
        # Skip 'Info_arr' on .tda files
        keys = keys[1:]

    if filetype in ("ts", "tda"):
        # remove time array from keys
        _ = keys.pop(0)

    return keys


def read_ts(name, ind=None, path='', verbose=False):
    """
    Read direct access formatted files with time series or cycle distributions

    Parameters
    ----------
    name : str
        Name of the binary file (incl. extension).
    ind : list of integers, optional
        Defines which responses to include in returned array. Each of the indices in `ind` refer the sequence number
        of the reponses. Note that `0` is the index of the time array.
    path : str, optional
        Path to file
    verbose : bool, optional
        Write info to screen?

    Returns
    -------
    array
        Array with time and responses.

    Notes
    -----
    If ``ind`` is specified, time array is only included if `0` is included in the specified indices.

    If ``ind`` is specified, the response arrays (1-D) are stored in the returned 2-D array in the same order as
    specified. I.e. if ``ind=[0,10,2,3]``, then the response array with index `10` on .ts file is obtained from
    ``arr[1,:]``.

    For reading of .dis files, keep in mind that there is no common time array or similar for each data set. For
    example; if one intends to request data set number three, one would need to request indices 4 and 5 (first set is
    stored on indices 0 and 1, second on 2 and 3, etc.)
    """
    # path
    tspath = os.path.join(path, name)
    if verbose:
        print('Reading %s ...' % name)
    # indices --> list or None
    if isinstance(ind, int):
        ind = [ind]
    # check extension
    ext = os.path.splitext(name)[-1]
    if ext in (".ts", ".dis"):
        fmt = "i"
    elif ext == ".tda":
        fmt = "f"
    else:
        raise ValueError("Invalid file format for this reader: %s" % ext)

    # The loop below, with try/except, is introduced to account for troubles
    # in reading .tda files (output from simo s2xmod). The following
    # combinations of platform and formats have been checked and found OK:
    #    .ts on Windows  : 'i', 'l'
    #    .ts on Linux    : 'i'
    #    .tda on Windows : 'f'
    #    .tda on Linux   : ??

    try:
        with open(tspath, 'rb') as f:
            # read ts-file information (1st line)
            nbytes = 4
            s = f.read(nbytes)
            ndat = unpack(fmt, s)[0]  # number of time steps per array

            # the code below works generally, but for some direct access files: nts=nrec-1
            # therefore, `nrec` is based on parsing the file instead
            #
            #   s = f.read(nbytes)
            #   nrec = unpack(fmt, s)[0]  # number of records(including info "row" and time array)
            #   nts = nrec-2

            f.seek(0, 2)  # go to end of file
            epos = f.tell()  # get end position (in bytes)
            nrec = epos / (ndat * nbytes)  # no. of "rows"
            nts = nrec - 2  # excl. time array
            # make integers (necessary for fmt='f')
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


