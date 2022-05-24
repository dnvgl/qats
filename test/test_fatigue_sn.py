# -*- coding: utf-8 -*-
"""
Module for testing module fatigue.sn
"""

import unittest
import numpy as np
from collections import OrderedDict, defaultdict
from scipy.optimize import brenth, brentq
from scipy.special import gamma
from qats.fatigue.sn import SNCurve, minersum, minersum_weibull

# todo: include tests for thickness correction of SNCurve class
# todo: include test for minersum() (fatigue minersum from stress range histogram)


class TestFatigueSn(unittest.TestCase):
    def setUp(self):
        """
        Common setup for all tests
        """
        # define sn curve parameters
        self.sndict_studless = dict( # single slope S-N curve, here: DNVGL-OS-E301 curve for studless chain
            name="Studless chain OS-E301",
            m1=3.0,
            a1=6e10,
        )
        self.sndict_B1_air = dict(  # two slope S-N curve, here: DNVGL-RP-C203 curve B1 in air
            name="B1 air",
            m1=4.0,
            m2=5.0,
            loga1=15.117,
            nswitch=1e7,
        )
        self.sndict_C_sea_cp = dict(  # two slope S-N curve, here: DNVGL-RP-C203 curve C in seawater with CP
            name="D seawater cp",
            m1=3.0,
            m2=5.0,
            loga1=12.192,
            nswitch=1e6,
        )
        # initiate sn curves
        self.sn_studless = SNCurve(**self.sndict_studless)
        self.sn_b1_air = SNCurve(**self.sndict_B1_air)
        self.sn_c_sea = SNCurve(**self.sndict_C_sea_cp)

    def test_sncurve_initiation_bilinear(self):
        """
        Test that correct S-N curve parameters are calculated at initiation of bilinear curve
        """
        b1_air = self.sn_b1_air
        c_sea = self.sn_c_sea
        # check that correct value of log(a2) is calculated at initiation
        self.assertAlmostEqual(b1_air.loga2, 17.146, places=3,
                               msg="SNCurve initiation, wrong value for log10(a2) (curve B1 air)")
        self.assertAlmostEqual(c_sea.loga2, 16.320, places=3,
                               msg="SNCurve initiation, wrong value for log10(a2) (curve C sea cp)")
        # check that correct value of Sswitch (or "fatigue limit") is calculated at initiation
        self.assertAlmostEqual(b1_air.sswitch, 106.97, places=2,
                               msg="SNCurve initiation, wrong value for sswitch (curve B1 air)")

    def test_sncurve_n(self):
        """
        Test that correct fatigue capacity (n) is calculated (using fatigue limit at 1e7 cycles, given in DNVGL-RP-C203
        tables 2-1 and 2-2.
        """
        b1_air = self.sn_b1_air
        c_sea = self.sn_c_sea
        self.assertAlmostEqual(np.log10(b1_air.n(106.967)), 7, places=5,
                               msg="Wrong fatigue capacity `n` calculated for S-N curve B1 air")
        # note: for C (sea, cp), RP-C203 says fatigue limit 73.10 at 1e7 cycles
        self.assertAlmostEqual(np.log10(c_sea.n(73.114)), 7, places=5,
                               msg="Wrong fatigue capacity `n` calculated for S-N curve C sea cp")

    def test_sncurve_fatigue_strength(self):
        """
        Test that correct fatigue limit is calculated (using fatigue limit at 1e7 cycles, given in DNVGL-RP-C203
        tables 2-1 and 2-2.
        """
        b1_air = self.sn_b1_air
        c_sea = self.sn_c_sea
        self.assertAlmostEqual(b1_air.fatigue_strength(1e7), 106.97, places=2,
                               msg="Wrong fatigue strength at 1e7 cycles calculated for S-N curve B1 air")
        # note: for C (sea, cp), RP-C203 says fatigue limit 73.10 at 1e7 cycles
        self.assertAlmostEqual(c_sea.fatigue_strength(1e7), 73.11, places=2,
                               msg="Wrong fatigue strength at 1e7 cycles calculated for S-N curve C sea cp")

    def test_sncurve_sswitch(self):
        """
        Test that attribute that fatigue limit at 'sswitch' is equal to 'nswitch'
        (to verify calculation of 'sswitch' at initiation)
        """
        b1_air = self.sn_b1_air
        c_sea = self.sn_c_sea
        self.assertAlmostEqual(b1_air.n(b1_air.sswitch), b1_air.nswitch, places=8,
                               msg="Wrong 'sswitch' calculated for S-N curve B1 air")
        self.assertAlmostEqual(c_sea.n(c_sea.sswitch), c_sea.nswitch, places=8,
                               msg="Wrong 'sswitch' calculated for S-N curve C sea cp")

    def test_minersum(self):
        """
        Test that correct fatigue Miner sum is calculated using bilinear S-N curve.
        """
        c_sea = self.sn_c_sea
        start, stop = c_sea.fatigue_strength(1e7), c_sea.fatigue_strength(1e5)
        srange = np.linspace(start, stop, 20)  # stress range histogram
        d = 0.5  # target damage
        count = np.array([c_sea.n(s) for s in srange]) / srange.size * d
        self.assertAlmostEqual(minersum(srange, count, c_sea), d, places=8,
                               msg="Wrong fatigue life (damage) from minersum()")

    def test_minersum_scf(self):
        """
        Test that correct fatigue Miner sum is calculated using bilinear S-N curve.
        """
        studless = self.sn_studless
        start, stop = studless.fatigue_strength(1e7), studless.fatigue_strength(1e5)
        srange = np.linspace(start, stop, 20)  # stress range histogram
        d = 0.5     # target damage (excl. SCF)
        scf = 1.15  # stress concentration factor
        d_scf = d * scf ** studless.m  # damage incl. SCF
        count = np.array([studless.n(s) for s in srange]) / srange.size * d
        self.assertAlmostEqual(minersum(srange, count, studless, scf=scf), d_scf, places=8,
                               msg="Wrong fatigue life (damage) from minersum() with SCF specified")

    def test_minersum_weibull_bilinear(self):
        """
        Test that correct fatigue Miner sum is calculated from Weibull stress range distribution.

        The test is performed as follows, for three different values of Weibull shape parameter:
            1. For each shape parameter; calculate scale parameter (q) of the equivalent Weibull distribution (i.e.
                Weib. dist. that gives specified fatigue life)
            2. Calculate fatigue damage (for one year) using specified shape parameter and calculated scale parameter.
            3. Compare calculated fatigue damage to fatigue life (damage) specified initially.
        """
        sn = self.sn_b1_air
        life = 100.
        dyear = 1 / life
        v0 = 0.1  # mean stress cycle frequency
        for h in (0.8, 1.0, 1.1):
            q = _q_calc(life, h, v0, sn)
            self.assertAlmostEqual(minersum_weibull(q, h, sn, v0, td=31536000), dyear, places=6,
                                   msg=f"Wrong fatigue life from minersum_weibull() for bilinear S-N curve and"
                                   f" shape={h}")

    def test_minersum_weibull_singleslope(self):
        """
        Test that correct fatigue Miner sum is calculated from Weibull stress range distribution.

        The test is performed as follows, for three different values of Weibull shape parameter:
            1. For each shape parameter; calculate scale parameter (q) of the equivalent Weibull distribution (i.e.
                Weib. dist. that gives specified fatigue life)
            2. Calculate fatigue damage (for one year) using specified shape parameter and calculated scale parameter.
            3. Compare calculated fatigue damage to fatigue life (damage) specified initially.
        """
        sn = self.sn_studless
        life = 100.
        dyear = 1 / life
        v0 = 0.1  # mean stress cycle frequency
        for h in (0.8, 1.0, 1.1):
            q = _q_calc_single_slope(life, h, v0, sn)
            self.assertAlmostEqual(minersum_weibull(q, h, sn, v0, td=31536000), dyear, places=6,
                                   msg=f"Wrong fatigue life from minersum_weibull() for linear S-N curve and shape={h}")

    def test_minersum_weibull_scf(self):
        """
        Test that SCF is correctly accounted for when fatigue damage is calculated from Weibull stress range
        distribution.
        """
        sn = self.sn_studless
        scf = 1.15
        life = 100.
        dyear_scf = (1 / life) * scf ** sn.m  # only correct for linear (single slope) S-N curves
        life_scf = life / scf ** sn.m
        v0 = 0.1  # mean stress cycle frequency
        h = 1.0
        q = _q_calc_single_slope(life, h, v0, sn)
        self.assertAlmostEqual(minersum_weibull(q, h, sn, v0, td=31536000, scf=scf), dyear_scf, places=6,
                               msg="SCF not correctly accounting for by minersum_weibull()")


def _q_calc(fatigue_life, h, v0, sn, method='brentq'):
    """
    Calculate Weibull scale parameter (q) that gives specified fatigue life using closed form expression
    in DNV-RP-C03 (2016) eq. F.12-1.

    Parameters
    ----------
    fatigue_life: float
        Fatigue life [years].
    h: float
        Weibull shape parameter (in 2-parameter distribution).
    v0: float,
        Cycle rate [1/s].
    sn: dict or SNCurve
        Dictionary with S-N curve parameters, alternatively an SNCurve instance.
        If dict, expected attributes are: 'm1', 'm2', 'a1' (or 'loga1'), 'nswitch'.
    method: str, optional
        Which root finding function to use. 'brentq': scipy.optimize.brentq, 'brenth': scipy.optimize.brenth

    Returns
    -------
    float
        Corresponding Weibull scale parameter.

    Notes
    -----
    If thickness correction was taken into account when establishing fatigue life, this is implicitly included in the
    scale parameter calculated. To obtain the scale parameter excl. thickness correction:
    >>> q_ = q_calc(fatigue_life, h, v0, sn)
    >>> q = q_ / (t / t_ref)**k
    where `t` is the thickness, `t_ref` is the reference thickness, and `k` is the thickness exponent.
    Keep in mind that ``t = t_ref`` if ``t < t_ref``.

    See Also
    --------
    q_calc_single_slope
    """
    rootfuncs = {
        'brentq': brentq,
        'brenth': brenth,
    }
    if method not in rootfuncs:
        raise ValueError("method must be either of: %s" % ', '.join(["'%s'" % k for k in rootfuncs.keys()]))

    if type(sn) not in (dict, OrderedDict, defaultdict, SNCurve):
        raise ValueError("`sn` must be dict-like or SNCurve instance")

    if not isinstance(sn, SNCurve):
        sn = SNCurve("", **sn)

    # fatigue life in seconds
    td = fatigue_life * 3600. * 24 * 365

    # calculate gamma parameters
    eps = np.finfo(float).eps  # machine epsilon
    func = rootfuncs[method]
    q = func(lambda qq: minersum_weibull(qq, h, sn, v0, td) - 1, a=eps, b=1e10)

    return q


# alternative to `q_calc`, utilizing that single-slope S-N curve is used
def _q_calc_single_slope(fatigue_life, h, v0, sn):
    """
    Calculate Weibull scale parameter (q) that gives specified fatigue life, for single-slope S-N curve.

    Parameters
    ----------
    fatigue_life: float
        Fatigue life [years].
    h: float
        Weibull shape parameter (in 2-parameter distribution).
    v0: float,
        Cycle rate [1/s].
    sn: dict or SNCurve
        Dictionary with S-N curve parameters, alternatively an SNCurve instance.
        If dict, expected attributes are: 'm1' and 'a1' (or 'loga1').
    Returns
    -------
    float
        Corresponding Weibull scale parameter.

    See Also
    --------
    q_calc
    """
    if type(sn) not in (dict, OrderedDict, defaultdict, SNCurve):
        raise ValueError("`sn` must be dict-like or SNCurve instance")

    if not isinstance(sn, SNCurve):
        sn = SNCurve("", **sn)

    # fatigue life in seconds
    td = fatigue_life * 3600. * 24 * 365

    # calculate q
    q = (v0 * td / sn.a1 * gamma(1 + sn.m1 / h)) ** (-1 / sn.m1)

    return q
