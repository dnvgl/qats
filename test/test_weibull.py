import unittest

from qats.stats.weibull import Weibull, pwm, mle, msm, lse, pwm2


# todo: more test cases for weibull class and functions


class WeibullTestCases(unittest.TestCase):
    """
    Test the general case where the loc parameter is non-zero.
    """
    def setUp(self):
        self.loc = 66000.
        self.scale = 410.
        self.shape = 2.

        wd = Weibull(loc=self.loc, scale=self.scale, shape=self.shape)
        self.x = wd.rnd(size=500, seed=3)

    def test_pwm(self):
        a, b, c = pwm(self.x)
        self.assertLessEqual((self.loc-a)/self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_msm(self):
        a, b, c = msm(self.x)
        self.assertLessEqual((self.loc-a)/self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_lse(self):
        a, b, c = lse(self.x)
        self.assertLessEqual((self.loc-a)/self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_mle(self):
        a, b, c = mle(self.x)
        self.assertLessEqual((self.loc-a)/self.loc, 0.1)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_fit(self):
        weib = Weibull.fit(self.x, method="pwm")
        a, b, c = pwm(self.x)
        self.assertEqual(weib.loc, a)
        self.assertEqual(weib.scale, b)
        self.assertEqual(weib.shape, c)


class Weibull2PTestCases(unittest.TestCase):
    """
    Test the case where the loc parameter is zero
    """
    def setUp(self):
        self.loc = 0.
        self.scale = 410.
        self.shape = 3.5

        wd = Weibull(loc=self.loc, scale=self.scale, shape=self.shape)
        self.x = wd.rnd(size=500, seed=3)

    def test_pwm2(self):
        b, c = pwm2(self.x)
        self.assertLessEqual((self.scale - b) / self.scale, 0.1)
        self.assertLessEqual((self.shape - c) / self.shape, 0.1)

    def test_fit_2p(self):
        weib = Weibull.fit(self.x, method="pwm2")
        b, c = pwm2(self.x)
        self.assertEqual(weib.loc, 0.)
        self.assertEqual(weib.scale, b)
        self.assertEqual(weib.shape, c)


if __name__ == '__main__':
    unittest.main()
