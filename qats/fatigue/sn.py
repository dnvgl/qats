#!/usr/bin/env python
# coding: utf-8
"""
Classes and functions for fatigue calculations:
    - SNCurve (class)
    - Fatigue damage calculation (functions)
"""
import numpy as np
from scipy.special import gamma as gammafunc, gammainc, gammaincc

# todo: Update SNCurve docstring to include description of class and attributes


class SNCurve(object):
    """
    S-N curve representing fatigue capacity versus cyclic stresses.

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
    t_exp : float, optional
        Thickness correction exponent. If not specified, thickness may not be specified in later calculations.
    t_ref : float, optional
        Reference thickness [mm]. If not specified, thickness may not be specified in later calculations.

    Attributes
    ----------
    a1 : float
        Intercept parameter for N <= `nswitch`.
    a2 : float
        Intercept parameter for N > `nswitch`. Equals `a1` for linear curves.
    loga1 : float
        Common logarithm with base 10 of `a1`.
    loga2 : float
        Common logarithm with base 10 of `a2`.
    m1 : float
        Negative inverse slope parameter (for bilinear curves: used for N < `nswitch`).
    m2 : float
        Negative inverse slope parameter for N > `nswitch`. Equals `m1` for linear curves.
    name : str
        S-N curve name.
    nswitch : float
        Number of cycles at transition from `m1` to `m2`. Applies only to bilinear curves.
    sswitch : float
        Stress range at transition from `m1` to `m2`. Applies only to bilinear curves.
    t_exp : float
        Thickness correction exponent.
    t_ref : float, optional
        Reference thickness [mm] for thickness correction.

    Notes
    -----
    For linear curves (single slope), the following input parameters are required: m1, a1 (or loga1).
    For bi-linear curves, the following parameters are required: m1, m2, nswitch, a1 (or loga1), t_exp, t_ref.

    If S-N curve is overdefined (e.g. both loga1 and a1 are defined), the S-N curve is established based on the
    parameter order listed above (under "Parameters").
    """

    def __init__(self, name, m1, **kwargs):
        self.name = name

        self.m1 = m1
        self.m2 = m2 = kwargs.get("m2", None)

        self.t_exp = t_exp = kwargs.get("t_exp", None)
        self.t_ref = t_ref = kwargs.get("t_ref", None)

        # check parameters
        if m1 is None:
            raise ValueError("parameter `m1` must be given")

        if t_exp is None and t_ref is None:
            # thickness correction not specified
            pass
        elif t_exp is None or t_ref is None:
            raise ValueError("if thickness correction is specified, both parameters `t_exp` and `t_ref` "
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
            sswitch = 10 ** ((loga1 - np.log10(nswitch)) / m1)
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
        """
        Returns True if S-N curve is bi-linear, otherwise False.
        """
        if self.m2 is None:
            return False
        else:
            return True

    @property
    def m(self):
        """
        Slope parameter of linear (single slope) S-N curve (equal to `m1`).
        Not available for bi-linear curves.
        """
        # should only be available for linear S-N curves - otherwise, m1 and m2 should be used!
        assert self.m2 is None, "For bi-linear curves, use `m1` and `m2` instead of `m`"
        return self.m1

    def fatigue_strength(self, n, t=None):
        """
        Magnitude of stress range leading to a particular fatigue life (in terms of number of cycles.

        Parameters
        ----------
        n : float
            Number of cycles (fatigue life) [-].
        t : float, optional
            Thickness [mm]. If specified, thickness reference and exponent must be defined for the S-N curve. If not
            specified, thickness correction is not taken into account.

        Returns
        -------
        float
            Fatigue strength, i.e. magnitude of stress range leading to specified fatigue life (no. of cycles).

        Raises
        ------
        ValueError: If thickness is specified, but thickness reference and exponent is not defined.
        """
        # thickness correction
        if t is not None:
            # thickness correction term
            try:
                tcorr = self.thickness_correction(t)
            except ValueError:
                raise
        else:
            # todo: consider to raise error if t_exp and t_ref is specified, but t not given (unless suppressed)
            # no thickness correction term implies tk=1.0
            tcorr = 1.0

        # S-N parameters for specified stress range
        if self.bilinear is False or n <= self.nswitch:
            m = self.m1
            loga = self.loga1
        else:
            m = self.m2
            loga = self.loga2

        # fatigue strength, ref. DNV-RP-C203 (2016) eq. 2.4.3
        s = 1 / tcorr * 10 ** ((loga - np.log10(n)) / m)

        return s

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
                tcorr = self.thickness_correction(t)
            except ValueError:
                raise
        else:
            # todo: consider to raise error if t_exp and t_ref is specified, but t not given (unless suppressed)
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

    def thickness_correction(self, t):
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
            If thickness correction is not defined, i.e. `t_exp` and `t_ref` are not defined.
        """
        try:
            if t < self.t_ref:  # t = tref is used for thickness less than tref, ref. DNV-RP-C203 eq. 2.4.3
                t = self.t_ref
            tcorr = (t / self.t_ref) ** self.t_exp
        except TypeError:
            raise ValueError("thickness correction is not defined, i.e. `t_exp` and `t_ref` are not specified")
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

        s += "t_exp : %(t_exp)s\n" \
             "t_ref   : %(t_ref)s\n" % self.__dict__

        print(s)
        return


def minersum(srange, count, sn, td=1., scf=1., th=None, retbins=False):
    """
    Fatigue damage (Palmgren-Miner sum) calculation based on stress cycle histogram and S-N curve.

    Parameters
    ----------
    srange: list of floats
        List of stress ranges in histogram (Note: only one value per bin).
    count: list of floats
        Cycle count for each of the stress ranges. May be specified as number of cycles [-] or cycle rate [1/s].
        If cycle rate is specified, specify duration `td` for scaling to number of cycles.
    sn: dict or SNCurve
        Dictionary with S-N curve parameters, alternatively an SNCurve instance.
        If dict is specified, expected keys are: 'm1' and 'a1' (or 'loga1') for linear S-N curve, and also 'm2' and
        'nswitch' if bi-linear S-N curve.
    td: float, optional
        Duration [s]. Used to scale the histogram from cycle rate to number of cycles.
        Use 1 (the default) if histogram already represents number of cycles.
    scf: float, optional
        Stress concentration factor to be applied on stress ranges. Default: 1.
    th: float, optional
        Thickness [mm] for thickness correction. If specified, reference thickness and thickness exponent must be
        defined for the S-N curve given.
    retbins: bool, optional
        If True, minersum per bin is also returned.

    Returns
    -------
    float
        Fatigue damage (Palmgren-Miner sum).
    list (optional)
        Fatigue damage (Palmgren-Miner sum) for each stress range bin. Returned if `retbin=True`.

    Raises
    ------
    ValueError:
        If thickness is given but thickness correction not specified for S-N curve.
    """
    if not len(srange) == len(count):
        raise ValueError("`srange` and `count` must be lists of same length")

    if not isinstance(sn, SNCurve):
        sn = SNCurve("", **sn)

    if th is not None and (sn.t_exp is None and sn.t_ref is None):
        raise ValueError("thickness is specified, but `k_tickn` and `t_ref` not defined for given S-N curve")

    damage_per_bin = [(td * n) / sn.n(s * scf, t=th) for s, n in zip(srange, count)]
    d = sum(damage_per_bin)

    if retbins is True:
        return d, damage_per_bin
    else:
        return d


def minersum_weibull(q, h, sn, v0, td=None, scf=1., th=None):
    """
    Fatigue damage (Palmgren-Miner sum) calculation based on (2-parameter) Weibull stress cycle distribution and
    S-N curve. Ref. DNV-RP-C03 (2016) eq. F.12-1.

    Parameters
    ----------
    q: float
        Weibull scale parameter (in 2-parameter distribution).
    h: float
        Weibull shape parameter (in 2-parameter distribution).
    sn: dict or SNCurve
        Dictionary with S-N curve parameters, alternatively an SNCurve instance.
        If dict, expected attributes are: 'm1', 'm2', 'a1' (or 'loga1'), 'nswitch'.
    v0: float,
        Cycle rate [1/s].
    td: float, optional
        Duration [s] (or design life, in seconds). Default is 31536000 (no. of seconds in a year, or 365 days).
    scf: float, optional
        Stress concentration factor to be applied on stress ranges.
    th: float, optional
        Thickness [mm] for thickness correction. If specified, reference thickness and thickness exponent must be
        defined for the S-N curve given.

    Returns
    -------
    float
        Fatigue damage (Palmgren-Miner sum).

    Raises
    ------
    ValueError:
        If thickness is given but thickness correction not specified for S-N curve.
    """
    def cigf(a, x):
        """ Complementary incomplete gamma function """
        return gammaincc(a, x) * gammafunc(a)

    def igf(a, x):
        """ Incomplete gamma function """
        return gammainc(a, x) * gammafunc(a)

    if not isinstance(sn, SNCurve):
        sn = SNCurve("", **sn)

    if td is None:
        td = 3600. * 24 * 365

    if th is not None:
        try:
            # include thickness correction in SCF
            scf *= sn.thickness_correction(th)
        except ValueError:
            raise

    # todo: verify implementation of thickness correction
    # scale Weibull scale parameter by SCF (incl. thickness correction if specified)
    q *= scf

    if sn.bilinear is True:
        # gamma functions
        g1 = cigf(1 + sn.m1 / h, (sn.sswitch / q) ** h)  # complementary incomplete gamma function
        g2 = igf(1 + sn.m2 / h, (sn.sswitch / q) ** h)   # incomplete gamma function
        # fatigue damage (for specified duration)
        d = v0 * td * (q ** sn.m1 / sn.a1 * g1 + q ** sn.m2 / sn.a2 * g2)
    else:
        # single slope S-N curve, fatigue damage for specified duration
        d = v0 * td * (q ** sn.m1 / sn.a1) * gammafunc(1 + sn.m1 / h)

    return d

