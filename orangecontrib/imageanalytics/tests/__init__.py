import os
import unittest


def suite(loader=unittest.TestLoader(), pattern='test*.py'):
    """Loads all project's tests."""
    dir_ = os.path.dirname(os.path.dirname(__file__))
    all_tests = loader.discover(dir_, pattern)
    return unittest.TestSuite(all_tests)


def load_tests(loader, tests, pattern):
    """Loads this module's tests."""
    module_dir = os.path.dirname(__file__)
    module_tests = loader.discover(module_dir, pattern)
    return unittest.TestSuite(module_tests)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
