"""
Testcases for the Needleman-Wunsch algorithm implementation.

Most results tests are based on the outputs of another software:
https://bioboot.github.io/bimm143_W20/class-material/nw/

@author: Bartosz Grabek
@date: 27-10-2024
"""

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

    def test_results_1_simple_multiple_paths(self):
        """This is an example result test verifying:
        - the score matrix used in the algorithm
        - the direction matrix used in the algorithm
        - the score of the alignment
        - the optimal paths
        """
        nw = NeedlemanWunsch(submatrix_file="data/submatrix2.csv", GP=-6)
        all_paths, score = nw.compare("TATA", "ATAT", n=2)
        expected_score_matrix = [
            [0, -6, -12, -18, -24],
            [-6, -2, -1, -7, -13],
            [-12, -1, -4, 4, -2],
            [-18, -7, 4, -2, 9],
            [-24, -13, -2, 9, 3],
        ]
        expected_dir_matrix = [
            ["0", "L", "L", "L", "L"],
            ["U", "D", "D", "L", "DL"],
            ["U", "D", "D", "D", "L"],
            ["U", "U", "D", "UL", "D"],
            ["U", "DU", "U", "D", "UL"],
        ]
        self.assertTrue(np.array_equal(nw.score_matrix, expected_score_matrix))
        self.assertEqual(nw.dir_matrix, expected_dir_matrix)
        self.assertEqual(score, 3)
        self.assertEqual(set(all_paths), set([("TATA-", "-ATAT"), ("-TATA", "ATAT-")]))

    def test_results_2_single_path_lab(self):
        """Test case based on example in the lab no.1 pdf"""
        nw = NeedlemanWunsch(submatrix_file="data/submatrix2.csv", GP=-6)
        all_paths, score = nw.compare("TGCTCGTA", "TTCATA", n=10)
        expected_score_matrix = np.array(
            [
                [0, -6, -12, -18, -24, -30, -36, -42, -48],
                [-6, 5, -1, -7, -13, -19, -25, -31, -37],
                [-12, -1, 3, -3, -2, -8, -14, -20, -26],
                [-18, -7, -3, 8, 2, 3, -3, -9, -15],
                [-24, -13, -9, 2, 6, 0, 1, -5, -4],
                [-30, -19, -15, -4, 7, 4, -2, 6, 0],
                [-36, -25, -21, -10, 1, 5, 2, 0, 11],
            ]
        )
        expected_dir_matrix = [
            ["0", "L", "L", "L", "L", "L", "L", "L", "L"],
            ["U", "D", "L", "L", "DL", "L", "L", "DL", "L"],
            ["U", "DU", "D", "DL", "D", "L", "L", "DL", "L"],
            ["U", "U", "DU", "D", "L", "D", "L", "L", "L"],
            ["U", "U", "DU", "U", "D", "DL", "D", "DL", "D"],
            ["U", "DU", "DU", "U", "D", "D", "DL", "D", "L"],
            ["U", "U", "DU", "U", "U", "D", "D", "U", "D"],
        ]
        self.assertTrue(np.array_equal(nw.score_matrix, expected_score_matrix))
        self.assertEqual(nw.dir_matrix, expected_dir_matrix)
        self.assertEqual(score, 11)
        self.assertEqual(set(all_paths), set([("TGCTCGTA", "T--TCATA")]))

    def test_results_3_multiple_paths(self):
        nw = NeedlemanWunsch(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = nw.compare("GATTACA", "GTCGACGCA", n=10)
        expected_score_matrix = np.array([
            [0, -2, -4, -6, -8, -10, -12, -14],
            [-2, 1, -1, -3, -5, -7, -9, -11],
            [-4, -1, 0, 0, -2, -4, -6, -8],
            [-6, -3, -2, -1, -1, -3, -3, -5],
            [-8, -5, -4, -3, -2, -2, -4, -4],
            [-10, -7, -4, -5, -4, -1, -3, -3],
            [-12, -9, -6, -5, -6, -3, 0, -2],
            [-14, -11, -8, -7, -6, -5, -2, -1],
            [-16, -13, -10, -9, -8, -7, -4, -3],
            [-18, -15, -12, -11, -10, -7, -6, -3]
        ])
        expected_dir_matrix = [['0', 'L', 'L', 'L', 'L', 'L', 'L', 'L'], ['U', 'D', 'L', 'L', 'L', 'L', 'L', 'L'], ['U', 'U', 'D', 'D', 'DL', 'L', 'L', 'L'], ['U', 'U', 'DU', 'D', 'D', 'DL', 'D', 'L'], ['U', 'DU', 'DU', 'DU', 'D', 'D', 'DL', 'D'], ['U', 'U', 'D', 'DU', 'DU', 'D', 'DL', 'D'], ['U', 'U', 'U', 'D', 'DU', 'U', 'D', 'L'], ['U', 'DU', 'U', 'DU', 'D', 'U', 'U', 'D'], ['U', 'U', 'U', 'DU', 'DU', 'DU', 'DU', 'DU'], ['U', 'U', 'DU', 'DU', 'DU', 'D', 'U', 'D']]
        self.assertTrue(np.array_equal(nw.score_matrix, expected_score_matrix))
        self.assertEqual(nw.dir_matrix, expected_dir_matrix)
        self.assertEqual(score, -3)
        self.assertEqual(set(all_paths), set([('GATTAC--A', 'GTCGACGCA'), ('GATTA--CA', 'GTCGACGCA')]))

if __name__ == "__main__":
    unittest.main()
