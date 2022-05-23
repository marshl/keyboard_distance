import unittest

from keyboard_distance.main import get_word_traversal_angle


class WordAngleTestCase(unittest.TestCase):
    def test_angle(self):
        full_angle, relative_angle = get_word_traversal_angle("the")
        self.assertEqual(full_angle, relative_angle)
        self.assertEqual(full_angle, 0.37624201063737334)
