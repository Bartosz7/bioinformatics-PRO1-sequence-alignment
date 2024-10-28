# tests/test_needleman_wunsch.py
import unittest
import sys
import os
import numpy as np

# Add the PRO1 directory to the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PRO1"))
)

from needleman_wunsch import NeedlemanWunsch
from smith_waterman import SmithWaterman


class TestNeedlemanWunsch(unittest.TestCase):

    def test_import_substitution_matrix(self):
        nw = NeedlemanWunsch(submatrix_file="data/submatrix.csv", GP=-2)
        expected = np.array(
            [[5, -4, -4, -1], [-4, -5, -4, -1], [-4, -4, 5, -1], [-1, -1, -1, 5]]
        )
        self.assertTrue(np.array_equal(nw.submatrix.to_numpy(), expected))

    def test_import_substitution_matrix_3(self):
        nw = NeedlemanWunsch(submatrix_file="data/submatrix3.csv", GP=-2)
        expected = np.array(
            [
                [10, -2, -2, -2, -2],
                [-2, 10, -2, -2, -2],
                [-2, -2, 10, -2, -2],
                [-2, -2, -2, 10, -2],
                [-2, -2, -2, -2, 10],
            ]
        )
        self.assertTrue(np.array_equal(nw.submatrix.to_numpy(), expected))

    def test_score_1(self):
        nw = NeedlemanWunsch(submatrix_file="data/submatrix.csv", GP=-2)
        self.assertEqual(nw.score("A", "A"), 5)
        self.assertEqual(nw.score("A", "C"), -4)
        self.assertEqual(nw.score("T", "T"), 5)
        self.assertEqual(nw.score("G", "T"), -1)
        self.assertEqual(nw.score("A", "A"), 5)

    def test_score_3(self):
        nw = NeedlemanWunsch(submatrix_file="data/submatrix3.csv", GP=-2)
        self.assertEqual(nw.score("A", "A"), 10)
        self.assertEqual(nw.score("A", "U"), -2)
        self.assertEqual(nw.score("T", "T"), 10)
        self.assertEqual(nw.score("G", "T"), -2)


if __name__ == "__main__":
    unittest.main()
