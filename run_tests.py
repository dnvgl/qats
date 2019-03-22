# -*- coding: utf-8 -*-
"""
Script for discovering and running unit tests
"""
import unittest
import sys

if __name__ == '__main__':
    test_suite = unittest.TestLoader().discover("test", "test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=1)
    test_result = test_runner.run(test_suite)
    sys.exit(not test_result.wasSuccessful())
