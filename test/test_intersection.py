import unittest

from keyboard_distance.__main__ import does_word_intersect_itself


class SimpleIntersectTestCase(unittest.TestCase):
    def test_non_intersecting(self):
        self.assertFalse(does_word_intersect_itself("qasw"))
        self.assertFalse(does_word_intersect_itself("qwer"))
        self.assertFalse(does_word_intersect_itself("qwerewq"))
        self.assertFalse(does_word_intersect_itself("qsderfvcxza"))

    def test_interesting(self):
        self.assertTrue(does_word_intersect_itself("qsaw"))