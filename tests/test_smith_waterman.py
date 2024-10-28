"""
Testcases for the Needleman-Wunsch algorithm implementation.

Most results tests are based on the outputs of another software:
https://bioboot.github.io/bimm143_W20/class-material/nw/
http://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Smith-Waterman

@author: Bartosz Grabek
@date: 27-10-2024
"""

# tests/test_smith_waterman.py
import unittest
import sys
import os
import numpy as np

# Add the PRO1 directory to the PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PRO1"))
)

from smith_waterman import SmithWaterman


class TestSmithWaterman(unittest.TestCase):

    def test_results_1_simple_multiple_paths_short_sequences(self):
        sw = SmithWaterman(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = sw.compare("AATCG", "AACG", n=10)
        expected_score_matrix = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 2, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 2],
            ]
        )
        expected_dir_matrix = [
            ["0", "L", "L", "L", "L", "L"],
            ["U", "D", "D", "", "", ""],
            ["U", "D", "D", "DL", "", ""],
            ["U", "", "DU", "D", "D", ""],
            ["U", "", "", "", "D", "D"],
        ]
        self.assertTrue(np.array_equal(sw.score_matrix, expected_score_matrix))
        self.assertEqual(sw.dir_matrix, expected_dir_matrix)
        self.assertEqual(score, 2)
        self.assertEqual(set(all_paths), set([("AA", "AA"), ("CG", "CG")]))

    def test_results_2_single_local_alignment_long_sequences(self):
        sw = SmithWaterman(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = sw.compare("AGTAGTGAGATGTGAGGGA", "GATAGAGGGGATGAAAATAG", n=10)
        self.assertEqual(score, 5)
        self.assertEqual(set(all_paths), set([("GAGGG", "GAGGG")]))

    def test_results_3_multiple_local_alignments_long_sequences(self):
        sw = SmithWaterman(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = sw.compare("ATGCTACTAGGTTAATACCTA", "CGTAGTGGCTGGCACCATGA", n=10)
        self.assertEqual(score, 3)
        self.assertEqual(set(all_paths), set([("ATG", "ATG"), ("ACC", "ACC"),
                                              ("GCT", "GCT"), ("TAG", "TAG")]))

    def test_results_4_single_local_alignment_long_sequences(self):
        sw = SmithWaterman(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = sw.compare("GATCGATCGATCCATA", "ATATCATCCAGGATAC", n=1)
        self.assertEqual(score, 6)
        self.assertEqual(set(all_paths), set([("ATCGATCCA", "ATC-ATCCA")]))

    def test_results_5_multiple_alignments_varying_length_sequences(self):
        sw = SmithWaterman(submatrix_file="data/submatrix4.csv", GP=-2)
        all_paths, score = sw.compare("ATATATCACTTACTCAT", "AGTCAGT", n=10)
        self.assertEqual(score, 3)
        self.assertEqual(set(all_paths), set([("TCA", "TCA"),
                                              ("TCACT", "TCAGT"),
                                              ("TCA", "TCA")]))


if __name__ == "__main__":
    unittest.main()
