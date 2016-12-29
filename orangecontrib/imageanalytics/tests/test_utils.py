from unittest import TestCase

from orangecontrib.imageanalytics.utils import get_hostname


class UtilsTest(TestCase):

    def test_get_hostname(self):
        self.assertEquals(get_hostname('http://example.com'), 'example.com')
        self.assertEquals(get_hostname('http://example.com/'), 'example.com')
        self.assertEquals(get_hostname('https://example.com'), 'example.com')
        self.assertEquals(get_hostname('http://127.0.0.1'), '127.0.0.1')
        self.assertEquals(get_hostname('http://s.example.com'), 's.example.com')
