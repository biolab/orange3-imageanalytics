from datetime import datetime
from unittest import TestCase


class TestUrllib(TestCase):
    def test_remove_urllib_fix(self):
        """
        When test start to fail check https://github.com/ionrock/cachecontrol/pull/294
        if already fixed and if fix cachecontrol also released. If true:
        - remove this test file
        - remove urllib3 pin (line 104 and 105)
        """
        self.assertLess(datetime.today(), datetime(2023, 5,30))