#!/usr/bin/env python
# coding: utf-8
"""
Classes and functions for fatigue calculations:
    - SNCurve (class)
    - Fatigue damage calculation (functions)
"""
import numpy as np

# todo: implement fatigue damage calculations (histogram, fitted distribution)
# todo: write unittest module for fatigue functions (SNClass, damage calculations, ...)


class SNCurve(object):
    """
    todo: Update SNCurve docstring to include description of class and attributes

    Attributes
    ----------
    name: str
    m1 : float
    m2 : float
    a1 : float
    a2 : float
    loga1 : float
    loga2 : float
    nswitch : float
    sswitch : float
    t_ref : float
    k_thickn : float
    """

    def __init__(self, name, m1, **kwargs):
        """

        Parameters
        ----------
        name : str
            S-N curve name
        m1 : float
            Negative inverse slope parameter (for bilinear curves: used for N < `nswitch`).
        m2 : float, optional
            Negative inverse slope parameter for N > `nswitch`
        a1 : float, optional
            Intercept parameter for N <= `nswitch`.
        loga1 : float, optional
            Log10 of a1. Must be given if `a1` is not specified.
        nswitch : float, optional
            Number of cycles at transition from `m1` to `m2`.
        k_thickn : float, optional
            Thickness correction exponent. If not specified, thickness may not be specified in later calculations.
        t_ref : float, optional
            Reference thickness [mm]. If not specified, thickness may not be specified in later calculations.

        Notes
        -----
        For linear curves (single slope), the following input parameters are required: m1, a1 (or loga1).
        For bi-linear curves, the following parameters are required: m1, m2, nswitch, a1 (or loga1), k_thickn, t_ref.

        If S-N curve is overdefined (e.g. both loga1 and a1 are defined), the S-N curve is established based on the
        parameter order listed above (under "Parameters").
        """
        self.name = name

        self.m1 = m1
        self.m2 = m2 = kwargs.get("m2", None)

        self.k_thickn = k_thickn = kwargs.get("k_thickn", None)
        self.t_ref = t_ref = kwargs.get("t_ref", None)

        # check parameters
        if m1 is None:
            raise ValueError("parameter `m1` must be given")

        if k_thickn is None and t_ref is None:
            # thickness correction not specified
            pass
        elif k_thickn is None or t_ref is None:
            raise ValueError("if thickness correction is specified, both parameters `k_thickn` and `t_ref` "
                             "must be specified")

        # check and deduct intercept parameter(s), etc.
        a1 = kwargs.get("a1", None)
        loga1 = kwargs.get("loga1", None)
        if a1 is not None:
            loga1 = np.log10(a1)
        elif loga1 is not None:
            a1 = 10 ** loga1
        else:
            raise ValueError("either `a1` or `loga1` must be specified")

        # handle bi-linear curves (
        if self.bilinear is True:
            nswitch = kwargs.get("nswitch", None)
            if nswitch is None:
                raise ValueError("`nswitch` must be specified for bi-linear curves")
            loga2 = m2 / m1 * loga1 + (1 - m2/m1) * np.log10(nswitch)
            a2 = 10 ** loga2
            sswitch = 10 ** ((loga1 - np.log10(nswitch)) / m1)  # todo: check formula for Sswitch
        else:
            a2 = None
            loga2 = None
            nswitch = None
            sswitch = None

        # store all parameters
        self.a1 = a1
        self.a2 = a2
        self.loga1 = loga1
        self.loga2 = loga2
        self.nswitch = nswitch
        self.sswitch = sswitch

    def __repr__(self):
        str_type = "bi-linear" if self.bilinear is True else "linear"
        return '<SNCurve "%s" (%s)>' % (self.name, str_type)

    @property
    def bilinear(self):
        """ Bool """
        if self.m2 is None:
            return False
        else:
            return True

    def n(self, s, t=None):
        """
        Predicted number of cycles to failure for specified stress range and thickness.

        Parameters
        ----------
        s : float
            Stress range [MPa].
        t : float, optional
            Thickness [mm]. If specified, thickness reference and exponent must be defined for the S-N curve. If not
            specified, thickness correction is not taken into account.

        Returns
        -------
        float
            Predicted number of cycles to failure.

        Raises
        ------
        ValueError: If thickness is specified, but thickness reference and exponent is not defined.
        """
        # thickness correction
        if t is not None:
            # thickness correction term
            try:
                tcorr = self.thickn_corr(t)
            except ValueError:
                raise
        else:
            # todo: consider to raise error if k_thickn and t_ref is specified, but t not given (unless suppressed)
            # no thickness correction term implies tk=1.0
            tcorr = 1.0

        # S-N parameters for specified stress range
        if self.bilinear is False or s >= self.sswitch:
            m = self.m1
            loga = self.loga1
        else:
            m = self.m2
            loga = self.loga2

        # fatigue limit, ref. DNV-RP-C203 (2016) eq. 2.4.3
        n = 10 ** (loga - m * np.log10(s * tcorr))

        return n

    def thickn_corr(self, t):
        """
        Thickness correction for specified thickness.

        Parameters
        ----------
        t : float
            Thickness [mm]

        Returns
        -------
        float
            Thickness correction factor.

        Raises
        ------
        ValueError
            If thickness correction is not defined, i.e. `k_thickn` and `t_ref` are not defined.
        """
        try:
            if t < self.t_ref:  # t = tref is used for thickness less than tref, ref. DNV-RP-C203 eq. 2.4.3
                t = self.t_ref
            tcorr = (t / self.t_ref) ** self.k_thickn
        except TypeError:
            raise ValueError("thickness correction is not defined, i.e. `k_thickn` and `t_ref` are not specified")
        return tcorr

    def print_parameters(self):
        s = "%(name)s\n" \
            "----------------------------------\n" \
            "m1      : %(m1).1f\n" \
            "a1      : %(a1).2e\n" \
            "log(a1) : %(loga1).3f\n" % self.__dict__
        if self.bilinear is True:
            s += "nswitch : %(nswitch).1e\n" \
                 "m2      : %(m2).1f\n" \
                 "a2      : %(a2).2e\n" \
                 "log(a2) : %(loga2).3f\n" \
                 "sswitch : %(sswitch).3f\n" % self.__dict__

        s += "k_tickn : %(k_thickn)s\n" \
             "t_ref   : %(t_ref)s\n" % self.__dict__

        print(s)
        return

