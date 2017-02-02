import os
import unittest


def load_tests(loader, tests, pattern):
    """Loads this module's tests."""
    module_dir = os.path.dirname(__file__)
    module_tests = loader.discover(module_dir, pattern)
    return unittest.TestSuite(module_tests)
