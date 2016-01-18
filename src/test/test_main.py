import unittest
import main

class TestMain(unittest.TestCase):
    def test_config_file_to_dict(self):
        d = main.parse_file_to_dict("../domains/dummy_config.txt")
        self.assertEqual(set(d.keys()), set(["k1", "k4", "k5", "k6", "k7"]))
        self.assertEqual(d["k1"], "v1")
        self.assertEqual(d["k4"], "v4")
        self.assertEqual(d["k5"], "v5")
        self.assertEqual(d["k6"], "v6")
        self.assertEqual(d["k7"], "v7")
