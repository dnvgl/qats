"""
Readers pickle dataframe formatted time series files
"""
import pandas as pd


def read_names(path):
    """
    Read time series names from a pickle dumps of dataframes.



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
    names = list(df)
    newnames = []
    for name in names:
        if isinstance(name, tuple):
            if len(name) == 2:
                newnames.append(name[0] + " " + name[1])
            if len(name) == 3:
                newnames.append(name[0] + " " + str(name[1]) + " " + name[2])

        else:
            newnames.append(name)
    return newnames


def read_data(path):
    """
    Read time series data arranged column wise on a dataframe pickle dump file.

    Dataframe multi indexed format, where the index is time:

    Object                  Sensor_1   ...                       Sensor_1
    Parameter               acc_x      ...                       acc_y
    0.0                     -1.361777  ...                       13.888657
    0.1                     -1.460476  ...                       13.888769
    0.2                     -1.550907  ...                       13.884339
    0.3                     -1.628700  ...                       13.872694
    0.4                     -1.690518  ...                       13.853050

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
    df.insert(0, "Time", df[df.keys()[0]].index.values)
    return df.T.to_numpy()
