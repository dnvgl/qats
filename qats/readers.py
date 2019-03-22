"""
Functions for reading various time series file formats.
"""
from struct import unpack
import os
import re
import fnmatch
import numpy as np
import h5py
from datetime import datetime, timedelta


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


def read_bin_keys(keyfile):
    """
    Read key-files for ts-files on format 'bin' (riflex dynmod).

    Parameters
    ----------
    keyfile : str
        Keyfile path

    Returns
    -------
    list
        List of key entries on key-file.

    Notes
    -----
    This function also works for generating list of keys for key-files
    belonging to ascii-files written from riflex dynmod.

    Warnings
    --------
    This function is under development, and does not presently handle all
    kinds of response listings. If an unknown response description is
    encountered, the key suffix will be `DOF<xx>`.

    """
    # Line id. for pure digit line entries
    linid = 'Lin'
    # zero padding width
    nlin = 2
    nseg = 3
    nnod = 3  # for elements/nodes

    # extract 'noddis', 'elmsfo', 'elmfor' (asc), ...
    keyfiletype = os.path.splitext(keyfile.split('_')[-1])[0].lower()

    # open and read
    with open(keyfile, 'r') as f:
        lines = f.readlines()

    # define parameters
    if keyfiletype == 'noddis':
        elkey = 'No'
    elif keyfiletype in ('elmsfo', 'elmfor'):
        elkey = 'El'
    else:
        elkey = ''
    suffices = _rifdyn_suffices(lines)

    # find start line for storage information
    i_start = 1 + lines.index(fnmatch.filter(lines, "*------------------------------------------------------*")[0])
    # extract storage info lines
    p = re.compile(r'ignore*')
    key_lines = [l for l in lines[i_start:] if l.strip() and not p.search(l)]

    # determine number of keys on each line, and in total
    nkeys = [int(line.split()[3]) for line in key_lines]
    nkeystot = sum(nkeys)

    # parse keys
    keys = [''] * nkeystot
    i = 0
    for nk, kl in zip(nkeys, key_lines):
        l = kl.split()
        # Line id.
        if l[0].isdigit():
            a = linid + str(l[0]).zfill(nlin)
        else:
            a = l[0]
        # Segment id.
        b = 'Seg' + str(l[1]).zfill(nseg)
        # El./Node id.
        c = elkey + str(l[2]).zfill(nnod)
        # define key entries
        for suff in suffices[:nk]:
            keys[i] = '_'.join([a, b, c, suff])
            i += 1
    return keys


def read_h5_keys(fn, path="", verbose=False):
    """
    Extract keys (data sets names) for `h5` data sets containing (or; interpreted as) time series.

    Parameters
    ----------
    fn: str
        File name.
    path: str, optional
        Path to location of `h5file`.
    verbose: bool, optional
        If True, print info.

    Returns
    -------
    list
        List of dset names (keys/addresses)
    """
    # todo: allow user to specify `kind`, to "force" how time series are interpreted.
    h5file = os.path.join(path, fn)
    if not os.path.isfile(h5file):
        raise FileNotFoundError("file not found: %s" % fn)

    if verbose:
        print('Identifying datasets on %s ...' % fn)

    dsetnames = []  # list of dataset names/keys (only for data sets that are time series)
    n_dsets = 0
    n_groups = 0

    with h5py.File(fn, "r") as f:
        allkeys = list(f.keys())
        for key in allkeys:
            if isinstance(f[key], h5py.Dataset):
                n_dsets += 1
                # check if dataset is a time series
                dset = f[key]
                timeinfo = _get_h5_timearray_info(dset)

                # include data set if it is interpreted as a time series (i.e. if timeinfo is not None)
                if timeinfo is not None:
                    dsetnames.append(key)

            elif isinstance(f[key], h5py.Group):
                n_groups += 1
                # print("group: ", key)
                for i in f[key]:
                    allkeys.append('{}/{}'.format(key, i))

            else:
                raise Exception("unexpected error: %s is not a dataset or a group" % key)

    if verbose:
        print("   no. of groups      : %4d" % n_groups + "   (incl. subgroups)")
        print("   no. of datasets    : %4d" % n_dsets)
        print("   no. of time series : %4d" % len(dsetnames) + "   (datasets interpreted as time series)")

    return dsetnames


def _rifdyn_suffices(txt):
    """
    Define response key suffices.

    The response suffices, as e.g. "Te", "Mx1", etc. are based on parsing the RIFLEX DYNMOD key-file, more specifically
    the lines containing "DOF" information.

    Parameters
    ----------
    txt : list of strings
        Each string is the line of a key-file.

    Returns
    -------
    list
        The suffices, e.g. 'Xd', 'Yd', or 'Te', 'Sy1', 'Sy2', etc.
    """
    # extract indices for blocks with DOF descriptions
    p = re.compile("following applies")
    inds = [txt.index(line) for line in [line for line in txt if p.search(line)]]

    # Always choose the last block:
    #   If there are two blocks, the first will be bar elements, for which only the axial tension is stored.
    #   This dof will also be the first dof for beam elements, if there are any. Hence, only the last block needs
    #   to be parsed.
    ind = inds[-1]
    p = re.compile("DOF")
    # list of DOF (...) strings
    dofstrings = [line for line in txt[ind:] if p.search(line)]

    # extract DOF descriptions by splitting on '='
    dofdescr = [x.split('=')[-1].strip() for x in dofstrings]

    # define suffices
    suffices = [''] * len(dofdescr)
    for i, ds in enumerate(dofdescr):
        # displacements
        if re.match('displacement', ds):
            # extract direction : x, y, z
            suff = ds.split()[2].upper() + 'd'
        # forces, moments
        elif re.match('Axial', ds):
            suff = 'Te'
        elif re.match('Torsional', ds):
            suff = 'Mx'
        elif re.match('Mom.', ds):
            dsp = ds.split()
            suff = 'M'
            # which axis
            suff += fnmatch.filter(dsp, '*axis*')[0][0]
            # which end
            suff += str(dsp[-1])
        elif re.match('Shear', ds):
            dsp = ds.split()
            suff = 'S'
            # which axis
            suff += fnmatch.filter(dsp, '*direction*')[0][0]
            # which end
            suff += str(dsp[-1])

        else:  # unknown description, use DOFxx
            suff = 'DOF' + str(i + 1).zfill(2)

        suffices[i] = suff

    return suffices


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


def read_bin(name, ind=None, path='', verbose=False):
    """
    Read binary ts-file on the format 'bin' by Riflex Dynmod.

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

    """
    # path
    tspath = os.path.join(path, name)
    if verbose:
        print('Reading %s ...' % name)

    # read
    nbytes = 4
    intfmt = 'i'  # integer format
    with open(tspath, 'rb') as f:
        # parse no. of records
        nbytesrec = unpack('i', f.read(4))[0]
        nrec = int(nbytesrec / nbytes + 2)  # plus 2 due to first and last cols.
        nts = int(nrec - 2 - 1)             # no. time series (excl. time array)

        # parse no. of time steps
        f.seek(0, 2)                        # go to end
        nbytestot = f.tell()                # get position
        f.seek(0)                           # go back to start
        ndat = int(nbytestot / (nbytes * nrec))

        # which keys/indices to extract
        if ind is None:
            ind = range(nts + 1)            # (including time array)

        # info
        if verbose:
            print('---------------------------------------')
            print('nts (no. of responses on file) : %d' % nts)
            print('ndat (no. of time steps)       : %d' % ndat)
            print('number of keys to read         : %d  (%s)' % (len(ind), "including time array index 0 is specified"))

        # prepare for reading
        basefmt = intfmt + 'f' + 'f' * nts + intfmt
        fmt = basefmt * ndat
        # read, unpack, reshape and transpose (ignoring first and last row
        s = f.read(nbytestot)
        arr = np.array(unpack(fmt, s)).reshape((ndat, nrec))[:, 1:-1].T

    # return specified indices
    return arr[ind, :]


def read_ascii(name, ind=None, path='', skiprows=None, verbose=False):
    """
    Read time series arranged column wise on ascii formatted file.

    Parameters
    ----------
    name : str
        Name of the binary file (incl. extension).
    ind : list or tuple, optional
        Which columns to read, with 0 being the first. For example, usecols = (1,4,5) will extract the 2nd, 5th and
        6th columns. The default, None, results in all columns being read.
    path : str, optional
        Path to file
    skiprows : int, optional
        Skip the first `skiprows` lines; default: 0.
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
    specified. I.e. if ``ind=[0,10,2,3]``, then the response array with index `10` on ascii file is obtained from
    ``arr[1,:]``.

    """
    # path
    tspath = os.path.join(path, name)
    if verbose:
        print('Reading %s ...' % name)

    # read
    with open(tspath) as f:
        # skip commmented lines at start of file
        pos = f.tell()
        while True:
            line = f.readline()
            if not line:
                # reached EOF
                raise IOError("reached EOF - uncommented lines not found")
            if line.startswith("#"):
                pos = f.tell()
                continue
            f.seek(pos)
            break
        # load arrays
        arr = np.loadtxt(f, skiprows=skiprows, usecols=ind, unpack=True)

    # info
    if verbose:
        nts, ndat = arr.shape
        print('---------------------------------------')
        print('nts (no. of responses read) : %d' % nts)
        print('ndat (no. of time steps)    : %d' % ndat)

    return arr


def read_h5(fn, dsetnames=None, path='', verbose=False):
    """
    Reader for time series stored on `.h5` (or `hdf5`) file format.


    Parameters
    ----------
    fn: str
        File name.
    dsetnames: str or list, optional
        List of dset names (keys/adresses) to desired data sets. If None (default), all (time series) data
        sets are read.
    path: str, optional
        Path to file location.
    verbose:  bool, optional
        Write info to screen?

    Returns
    -------
    list
        List of arrays with time and data

    Notes
    -----
    If `dsetnames` is not specified, all datasets that are interpreted as time series will be read. See also
    `read_h5_keys`.

    hdf5 is an extremely flexible file format, so the read success, and extraction of time series, only relies on
    correct interpretation of the file structure. Currently, the following .h5 formats are implemented:

    - .h5 exported from SIMA

    """
    h5file = os.path.join(path, fn)
    if not os.path.isfile(h5file):
        raise FileNotFoundError("file not found: %s" % fn)

    if verbose:
        print('Reading %s ...' % fn)

    if isinstance(dsetnames, str):
        dsetnames = [dsetnames]
    elif type(dsetnames) in (list, tuple):
        pass
    elif dsetnames is None:
        pass
    else:
        raise TypeError("`dsetnames` must be str/list/tuple, got: %s" % type(dsetnames))

    # if dsetnames not specified, get all keys
    if dsetnames is None:
        dsetnames = read_h5_keys(h5file)

    arrays = []

    with h5py.File(fn, "r") as f:
        for key in dsetnames:
            dset = f[key]

            if not isinstance(dset, h5py.Dataset):
                raise TypeError("expected to get h5py.Dataset, got %s (for key '%s')" % (type(dset), key))

            # get values (data array), check size and dimensions
            data = dset[:]  # dset.value
            # todo: consider if check of data type is really neccessary -- if not, dset.ndim, dset.size etc. may be used
            if not isinstance(data, np.ndarray):
                raise NotImplemented("only value of type np.ndarray is implented, got: %s (for key '%s')" %
                                     (type(data), key))
            if data.ndim != 1:
                raise NotImplemented("only 1-dimensional arrays implemented, got ndim=%d (for key '%s')" %
                                     (data.ndim, key))
            data = data.flatten()   # flatten data array
            nt = data.size          # number of time steps

            # attrs = dset.attrs  # dict with dataset attributes

            # establish or extract time array
            timeinfo = _get_h5_timearray_info(dset)
            if timeinfo is None:
                raise Exception("no time info extracted for dataset '%s'" % key)
            kind = timeinfo["kind"]
            if kind == "sima":
                t_start = timeinfo["start"]
                dt = timeinfo["dt"]
                t_end = t_start + (nt-1) * dt
                timearr, step = np.linspace(t_start, t_end, nt, retstep=True)
                if not dt == step:
                    raise Exception("unexpected error: `dt` should be %s but is %s" % (dt, step))
            elif False:
                # todo: expand when _get_h5_timearray_info() has been expanded
                pass
            # verify that time array has correct shape (==> should be same as `data` shape)
            if not timearr.shape == data.shape:
                raise Exception("unexpected error: `time` has shape " + str(timearr.shape) + "while data has"
                                "shape " + str(data.shape) + " (should be equal)")
            # make 2-dimensional array with time and data, and append to output
            arr = np.array([timearr, data])
            arrays.append(arr)

    return arrays


def _get_h5_timearray_info(dset):
    """

    Parameters
    ----------
    dset: h5py.Dataset
        Dataset to check/extract time array info on.

    Returns
    -------
    dict
        Dictionary with time array info, keys.
        Returns None if time array info is not found/understood.
    """
    if not isinstance(dset, h5py.Dataset):
        raise TypeError("expected h5py.Dataset, got: %s" % type(dset))

    timeinfo = None

    attrs = dset.attrs
    if "start" in attrs and "delta" in attrs:
        # SIMA-way of defining time array
        timeinfo = {
            "kind": "sima",
            "start": attrs["start"],
            "dt": attrs["delta"],
        }
    elif False:
        # todo: expand criteria list to support h5 files other than those exported from SIMA
        pass

    return timeinfo


def _matlab_datenums_to_datetime(timearr):
    """
    Convert array of MATLAB datenum floats to array of datetime objects.

    Parameters
    ----------
    timearr: array_like or float

    Returns
    -------
    array
        Array of datetime objects, same shape as input array.
    """
    def convert(dn):
        # ref: https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
        _dtg = datetime.fromordinal(int(dn)) + timedelta(days=dn % 1) - timedelta(days=366)
        return _dtg

    # return quickly if float (not array) is given
    if np.ndim(timearr) == 0:
        return convert(timearr)

    # convert array
    timearr = np.asarray(timearr)
    was_shape = timearr.shape
    timearr = timearr.flatten()
    dtarray = np.array([convert(t) for t in timearr])
    return dtarray.reshape(was_shape)
