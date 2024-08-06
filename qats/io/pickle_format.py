"""
Readers pickle dataframe formatted time series files
"""
import numpy as np
import pandas as pd


def read_pickle_names(path):
    """
    Read time series names from a pickle dumps of dataframes.
    Converts multi index tuples to strings.

    Parameters
    ----------
    path : str
        Name of .pkl file

    Returns
    -------
    list
        Time series names


    """
    df = pd.read_pickle(path)
    if isinstance(df, pd.DataFrame):
        newnames = []
        for name in df.columns:
            if isinstance(name, tuple):
                # Convert tuple to string
                newnames.append(" ".join(str(item) for item in name))
            else:
                newnames.append(name)
    else:
        raise ValueError("Input file does not contain a pandas dataframe.")
    return newnames


def read_data(path):
    """
    Read time series data arranged column wise on a dataframe pickle dump file.

    Dataframe with data in single- or multiindexed columns, where the index is
    the common time array. Example (w/multiindexed columns):

    Object                  Sensor_1   ...                       Sensor_1
    Parameter               acc_x      ...                       acc_y
    0.0                     -1.361777  ...                       13.888657
    0.1                     -1.460476  ...                       13.888769
    0.2                     -1.550907  ...                       13.884339
    0.3                     -1.628700  ...                       13.872694
    0.4                     -1.690518  ...                       13.853050
    ...

    Parameters
    ----------
    path : str
        Pickle file path (relative or absolute)
    Returns
    -------
    array
        Time and data

    """
    df = pd.read_pickle(path)
    if isinstance(df, pd.DataFrame):
        df.insert(0, "Time", df.index.values)
    else:
        raise ValueError("Input file is not a pandas dataframe.")
    return df.T.to_numpy()


def write_data(path, time: np.ndarray, data: dict):
    """
    Construct dataframe and write to pickle file.

    Parameters
    ----------
    path : str
        File path
    time : array
        Time (common time array).
    data : dict
        Time series data in dict with name as key, and tuples with time and data arrays as values.
    """
    # construct dict with data arrays only (remove time array from each dict entry)
    data = {k: v[1] for k, v in data.items()}

    # construct dataframe and replace index by common time array
    df = pd.DataFrame(data)
    df.index = time

    # export to pickle
    df.to_pickle(path)

    return




