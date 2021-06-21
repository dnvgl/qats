#!/usr/bin/env python
# coding: utf-8
"""
Classes and functions for fatigue capacity and fatigue damage calculations
"""
import numpy as np
from scipy.special import gamma as gammafunc, gammainc, gammaincc
from typing import Optional, Tuple, Union


class SNCurve(object):
    """
    S-N data model

    Attributes
    ----------
    m : float
        Slope of first leg of the curve
    b0: float
        Constant coefficient in equation to calculate the intercept parameter
    name : Optional[str]
        Name/identifier
    description : Optional[str]
        Description of the curve
    unit: Optional[str]
        Unit of measure for the stress ranges e.g. MPa
    b1: Optional[float]
        Mean stress coefficient in equation to calculate the intercept parameter
    b2: Optional[float]
        Corrosion grade coefficient in equation to calculate the intercept parameter
    default_g1: Optional[float]
        Default value of g1(s_m) function, where s_m is the mean load
    default_g2: Optional[float]
        Default value of g2(c) function, where c is the corrosion grade
    m2: Optional[float]
        Slope of second leg of the curve, applies only to bi-linear curves
    n_switch: Optional[float]
        Point (in terms of number of cycles to failure) where the slope changes, applies only to bi-linear curves
    fatigue_limit: Optional[float]
        Fatigue limit in terms of stress range
    t_ref: Optional[float]
        Reference thickness for thickness correction
    t_exp: Optional[float]
        Exponent for thickness correction

    Notes
    -----
    The dependency to mean load and corrosion is included by expressing the intercept parameter as a
    function of these parameters:

        logA(s_m, c) = b0 + b1 * g1(s_m) + b2 * g2(c)

    where b0, b1 and b2 are constant coefficients and g1 and g2 are functions of mean load (s_m) and
    corrosion grade (c) respectively. These functions are generally nonlinear.

    References
    ----------
    1. Lone et.al. (2021), Fatigue assessment of mooring chain considering the effects of mean load and corrosion,
       OMAE2021-62775

    """
    def __init__(self, m: float, b0: float, name: str = None, description: str = None, unit: str = None,
                 b1: float = None, b2: float = None, default_g1: float = None, default_g2: float = None,
                 m2: float = None, n_switch: float = None, fatigue_limit: float = None, t_ref: float = None,
                 t_exp: float = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.m = m
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.default_g1 = default_g1
        self.default_g2 = default_g2
        self.m2 = m2
        self.n_switch = n_switch
        self.fatigue_limit = fatigue_limit
        self.t_ref = t_ref
        self.t_exp = t_exp

    def is_bilinear(self) -> bool:
        """Is it a bi-linear curve."""
        return self.m2 is not None and self.n_switch is not None

    def is_mean_load_and_corrosion_grade_dependent(self) -> bool:
        """Is it a curve which depends on mean load and corrosion grade."""
        return self.b1 is not None and self.b2 is not None and \
            self.default_g1 is not None and self.default_g2 is not None

    def loga(self, g1: Union[float, np.ndarray] = None, g2: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        Logarithm of intercept parameter

        Parameters
        ----------
        g1 : float or np.ndarray, optional
            value of g1(s_m), where s_m is the mean load
        g2 : float or np.ndarray, optional
            value of g1(c), where c is the corrosion grade

        Returns
        -------
        float or np.ndarray
            Logarithm of intercept parameter

        Notes
        -----
        The specified mean load (g1) and corrosion grade (g2) must be consistent with the parameter
        definitions applied with the actual S-N curve.

        For curves that are mean load and corrosion grade dependent, the curves
        reference values are applied by default if not provided.
        """
        if self.is_mean_load_and_corrosion_grade_dependent():
            if g1 is None:
                g1 = self.default_g1

            if g2 is None:
                g2 = self.default_g2

            return self.b0 + self.b1 * g1 + self.b2 * g2

        else:
            return self.b0

    def a(self, g1: Union[float, np.ndarray] = None, g2: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """
        Intercept parameter

        Parameters
        ----------
        g1 : float or np.ndarray, optional
            value of g1(s_m), where s_m is the mean load
        g2 : float or np.ndarray, optional
            value of g1(c), where c is the corrosion grade

        Returns
        -------
        float or np.ndarray
            Intercept parameter

        Notes
        -----
        The specified mean load (g1) and corrosion grade (g2) must be consistent with the parameter
        definitions applied with the actual S-N curve.

        For curves that are mean load and corrosion grade dependent, the curves
        reference values are applied by default if not provided.
        """
        return 10. ** self.loga(g1=g1, g2=g2)

    def loga2(self) -> float:
        """
        Logarithm of intercept parameter for the second leg.

        Applies only to bi-linear curves.
        """
        assert self.is_bilinear(), "The second intercept parameter applies only to bi-linear S-N curves. " \
                                   "Make sure that both `m2` and `n_switch` are specified."
        return self.m2 / self.m * self.loga() + (1 - self.m2/self.m) * np.log10(self.n_switch)

    def a2(self) -> float:
        """
        Intercept parameter for the second leg.

        Applies only to bi-linear curves.
        """
        return 10. ** self.loga2()

    def s_switch(self) -> float:
        """The stress range where the slope changes, applies only to bi-linear curves"""
        assert self.is_bilinear(), "The second intercept parameter applies only to bi-linear S-N curves. " \
                                   "Make sure that both `m2` and `n_switch` are specified."
        return 10 ** ((self.loga() - np.log10(self.n_switch)) / self.m)

    def thickness_correction(self, t: float) -> float:
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
        assert self.t_ref is not None and self.t_exp is not None, "Thickness correction parameters " \
                                                                  "`t_ref` and `t_exp` are not defined."

        if t < self.t_ref:  # t = tref is used for thickness less than tref, ref. DNV-RP-C203 eq. 2.4.3
            t = self.t_ref

        return (t / self.t_ref) ** self.t_exp

    def strength(self, n: Union[float, np.ndarray], g1: Union[float, np.ndarray] = None,
                 g2: Union[float, np.ndarray] = None, t: float = None) -> Union[float, np.ndarray]:
        """
        Magnitude of stress range leading to a particular fatigue life (in terms of number of cycles.

        Parameters
        ----------
        n : float or numpy.ndarray
            Number of cycles (fatigue life) [-].
        g1 : float or numpy.ndarray, optional
            value of g1(s_m), where s_m is the mean load
        g2 : float or numpy.ndarray, optional
            value of g1(c), where c is the corrosion grade
        t : float, optional
            Thickness [mm]. If specified, thickness reference and exponent must be defined for the S-N curve. If not
            specified, thickness correction is not taken into account.

        Returns
        -------
        float or numpy.ndarray
            Magnitude of stress range leading to specified fatigue life (no. of cycles).

        Notes
        -----
        The specified mean load (g1) and corrosion grade (g2) must be consistent with the parameter
        definitions applied with the actual S-N curve.

        For curves that are mean load and corrosion grade dependent, the curves
        reference values are applied by default if not provided.

        """
        # TODO: Handle cases where a fatigue limit is specified, np.nan for n > onset of the fatigue limit?
        n = np.asarray(n)
        g1 = np.asarray(g1)
        g2 = np.asarray(g2)

        # thickness correction
        if t is not None:
            t_correction = self.thickness_correction(t)
        else:
            t_correction = 1.0

        # S-N parameters for specified stress range
        if not self.is_bilinear() or n <= self.n_switch:
            m = self.m
            loga = self.loga(g1=g1, g2=g2)
        else:
            m = self.m2
            loga = self.loga2()

        # fatigue strength, ref. DNV-RP-C203 (2016) eq. 2.4.3
        s = 1. / t_correction * 10 ** ((loga - np.log10(n)) / m)

        # return float if input `n` was a float
        if s.ndim == 0:
            # float
            return s.item()
        else:
            return s

    def n(self, s: Union[float, np.ndarray], g1: Union[float, np.ndarray] = None, g2: Union[float, np.ndarray] = None,
          t: float = None) -> Union[float, np.ndarray]:
        """
        Predicted number of cycles to failure for specified stress range(s).

        Parameters
        ----------
        s : float or numpy.ndarray
            Stress range(s).
        g1 : float or numpy.ndarray, optional
            value of g1(s_m), where s_m is the mean load
        g2 : float or numpy.ndarray, optional
            value of g1(c), where c is the corrosion grade
        t : float, optional
            Thickness [mm]. If specified, thickness reference and exponent must be defined for the S-N curve. If not
            specified, thickness correction is not taken into account.

        Returns
        -------
        float or numpy.ndarray
            Predicted number of cycles to failure. Output type is same as input type (float or np.ndarray)

        """
        # TODO: Handle cases where a fatigue limit is specified, np.inf for s < fatigue limit?
        s = np.asarray(s)
        g1 = np.asarray(g1)
        g2 = np.asarray(g2)
        # thickness correction
        if t is not None:
            t_correction = self.thickness_correction(t)
        else:
            t_correction = 1.0

        # calculate number of cycles to failure `n`, ref. DNV-RP-C203 (2016) eq. 2.4.3
        # ... using appropriate S-N parameters for specified stress range(s)
        if not self.is_bilinear():
            # single slope sn curve
            return 10 ** (self.loga(g1=g1, g2=g2) - self.m * np.log10(s * t_correction))
        else:
            # bi-linear curve
            n = np.zeros(s.shape)

            # find stress ranges above the point where the slope changes
            ind = s >= self.s_switch()
            # fatigue limits for upper part of curve
            n[ind] = 10 ** (self.loga(g1=g1, g2=g2) - self.m * np.log10(s[ind] * t_correction))

            # fatigue limits for lower part of curve
            n[~ind] = 10 ** (self.loga2() - self.m2 * np.log10(s[~ind] * t_correction))

            # return float if input `s` was a float
            if n.ndim == 0:
                # float
                return n.item()
            else:
                return n

    def minersum(self, ranges: np.ndarray, count: np.ndarray, g1: Union[float, np.ndarray] = None,
                 g2: Union[float, np.ndarray] = None, df: float = 1., scf: float = 1., th: float = None
                 ) -> Tuple[float, np.ndarray]:
        """
        Fatigue damage calculation (Miner-Palmgren sum) based on stress cycle histogram

        Parameters
        ----------
        ranges : np.ndarray
            Stress cycle ranges
        count : np.ndarray
            Stress cycle counts
        g1 : float or numpy.ndarray, optional
            value of g1(s_m), where s_m is the mean load
        g2 : float or numpy.ndarray, optional
            value of g1(c), where c is the corrosion grade
        df: float, optional
            Factor scaling reference duration. Can be used to scale the histogram from cycle rate to number of cycles or
            to scale from one reference duration say. 100 seconds to a different reference duration say 10800 seconds.
            Use 1 (the default) if histogram already represents number of cycles during the same period of time.
        scf: float, optional
            Stress concentration factor to be applied on stress ranges. Default: 1.
        th: float, optional
            Thickness [mm] for thickness correction. If specified, reference thickness and thickness exponent must be
            defined for the S-N curve given.

        Returns
        -------
        float
            Total fatigue damage (Palmgren-Miner sum).
        np.ndarray
            Fatigue damage (Palmgren-Miner sum) per bin in the cycle distribution

        """
        # calculate fatigue damage per bin (combination of range and count) of the cycle distribution
        #  apply mean load and corrosion grade dependency if relevant (kwargs)
        #  scale stresses with stress concentration factor
        #  scale to the correct reference duration
        #  apply thickness correction if any
        damage_per_bin = df * count / self.n(s=ranges * scf, g1=g1, g2=g2, t=th)
        return float(np.sum(damage_per_bin)), damage_per_bin

    def minersum_weibull(self, q: float, h: float, v0: float, duration: float = 31556952., scf: float = 1.,
                         th: float = None):
        """
        Fatigue damage (Palmgren-Miner sum) calculation based on (2-parameter) Weibull stress cycle distribution and
        S-N curve. Ref. DNV-RP-C03 (2016) eq. F.12-1.
    
        Parameters
        ----------
        q: float
            Weibull scale parameter (in 2-parameter distribution).
        h: float
            Weibull shape parameter (in 2-parameter distribution).
        v0: float,
            Cycle rate [1/s].
        duration: float, optional
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

        if th is not None:
            try:
                # include thickness correction in SCF
                scf *= self.thickness_correction(th)
            except ValueError:
                raise

        # todo: verify implementation of thickness correction
        # scale Weibull scale parameter by SCF (incl. thickness correction if specified)
        q *= scf

        if self.is_bilinear():
            # gamma functions
            g1 = cigf(1 + self.m / h, (self.s_switch() / q) ** h)  # complementary incomplete gamma function
            g2 = igf(1 + self.m2 / h, (self.s_switch() / q) ** h)   # incomplete gamma function
            # fatigue damage (for specified duration)
            d = v0 * duration * (q ** self.m / self.a() * g1 + q ** self.m2 / self.a2() * g2)
        else:
            # single slope S-N curve, fatigue damage for specified duration
            d = v0 * duration * (q ** self.m / self.a()) * gammafunc(1 + self.m / h)

        return d
