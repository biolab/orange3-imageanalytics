import os
import unittest


def load_tests(loader, tests, pattern):
    """Loads this module's tests."""
    module_dir = os.path.dirname(__file__)
    top_level = os.path.realpath(
        os.path.join(module_dir, "..", "..", "..", "..")
    )
    module_tests = loader.discover(module_dir, pattern or "test*.py",
                                   top_level_dir=top_level)
    return unittest.TestSuite(module_tests)
