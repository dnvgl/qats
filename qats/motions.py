#!/usr/bin/env python
# coding: utf-8
"""
Transformations and operations related to motion.
"""
import numpy as np


def transform_motion(motion, newref, rotunit="deg"):
    """
    Transform motion to new reference position.

    The following sequence of rotation is used: z-y-z (known as yaw-pitch-roll). For more detailed description,
    see https://en.wikipedia.org/wiki/Euler_angles or SIMO Theory Manual (ch. 6.2 in version 4.14.0).

    Parameters
    ----------
    motion : tuple or np.ndarray
        Motion of old reference point, 6 dofs (hence shape (6, nt), where nt is number of time steps):
            - x (position of old reference point in global coord. system)
            - y (position of old reference point in global coord. system)
            - z (position of old reference point in global coord. system)
            - rx (rotation about local x-axis, aka. roll)
            - ry (rotation about local x-axis, aka. roll)
            - rz (rotation about local x-axis, aka. roll)
    newref : 3-tuple
        Position vector (in body coordinate system) of new reference point: (x, y, z).
    rotunit : {'deg', 'rad'}
        Unit of rotations. Default is degrees.

    Returns
    -------
    xyz : np.ndarray
        Position vector for new reference position, shape (3, n).
    """
    motion = np.asarray(motion)  # ensure motion is numpy array
    newref = np.asarray(newref)  # ensure newref is numpy array, for efficiency in loop with np.dot
    ndof, nt = motion.shape      # number of dofs and time steps
    assert ndof == 6, f"Motion must be of shape (6, nt) (6-dof motion), got {motion.shape}"
    assert newref.size == 3, f"Specified position must be list/tuple with three values, got {len(newref)}"

    # extract rotations, scale to radians if needed given as degrees
    if rotunit == "deg":
        rx, ry, rz = np.radians(motion[-3:, :])
    elif rotunit == "rad":
        rx, ry, rz = motion[-3:, :]
    else:
        raise ValueError(f"Parameter `rotunit` must be 'deg' or 'rad', not '{rotunit}'")

    # calculate transformation matrix -> shape (3, 3, nt)
    trans = np.array([
        [np.cos(rz) * np.cos(ry), -np.sin(rz) * np.cos(rx) + np.cos(rz) * np.sin(ry) * np.sin(rx),
         np.sin(rz) * np.sin(rx) + np.cos(rz) * np.sin(ry) * np.cos(rx)],
        [np.sin(rz) * np.cos(ry), np.cos(rz) * np.cos(rx) + np.sin(rz) * np.sin(ry) * np.sin(rx),
         -np.cos(rz) * np.sin(rx) + np.sin(rz) * np.sin(ry) * np.cos(rx)],
        [-np.sin(ry), np.cos(ry) * np.sin(rx), np.cos(ry) * np.cos(rx)],
    ])

    # transform to new reference position for each time step -> shape (3, nt)
    xyz = np.zeros((3, nt))
    for i in range(nt):
        xyz[:, i] = np.dot(trans[:, :, i], newref)
    # ... and add xyz motion of old reference point
    xyz += motion[:3, :]

    return xyz


def acceleration(x, t):
    """
    Numerical time differentiation to obtain acceleration of signal(s).


    Parameters
    ----------
    x : list or np.array
        Signal(s) to perform time derivation on. See requirements given for function :func:`velocity`.
    t : float or np.array
        Time step or time array.

    Returns
    -------
    acc : np.array
        Acceleration of input signal(s); same shape as input.

    Notes
    -----
    In practice, acceleration is calculated by calling function `velocity` twice.

    See Also
    --------
    velocity
    numpy.gradient
    """
    acc = x[:]  # copy input signals
    for _ in range(2):
        acc = velocity(acc, t)
    return acc


def velocity(x, t):
    """
    Numerical time differentiation to obtain velocity of signal(s).

    Parameters
    ----------
    x : list or np.array
        Signal to perform time derivation on. If a 2-D array is given, it is required to be of shape (n, nt), where `n`
        is number of signals and `nt` is number of time steps. Time differentiation is then performed for each of the
        signals independently.
    t : float or np.array
        Time step or time array.

    Returns
    -------
    vel : np.array
        Velocity of input signal(s); same shape as input.

    Notes
    -----
    Velocity is calculated using second order central differences, with first order backwards and forward differences
    at start/end of signals.
    See `numpy.gradient <https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html>`_.


    See Also
    --------
    numpy.gradient

    """
    # check input signal(s)
    x = np.asarray(x)
    assert x.ndim in (1, 2), "Input signal 'x' must be 1- or 2-D array"
    # check specified time
    if isinstance(t, float):
        # time step is specified
        pass
    else:
        # assume time array is specified -> check that it matches the number of time steps in input signals
        t = np.asarray(t)   # ensure t is numpy array
        nt = x.shape[-1]    # number of time steps
        assert t.size == nt, "If time array is specified, it must match number of time steps in input signal(s)"

    if x.ndim == 1:
        axis = None
    else:
        # more than one signal is given -> perform time differentiation along axis 1
        axis = 1

    # time differentiation
    vel = np.gradient(x, t, axis=axis)

    return vel

