"""
Smith-Waterman (SW) algorithm for comparison of two sequences (local alignment).
Implemented as an adaptation of the Needleman-Wunsch algorithm.

@author: Bartosz Grabek
@date: 27-10-2024
"""

import numpy as np
import pandas as pd
from needleman_wunsch import NeedlemanWunsch


class SmithWaterman(NeedlemanWunsch):
    """Implements the Smith-Waterman algorithm
    for two sequence alignment. Reuses the Needleman-Wunsch code
    through inheritance.

    Note: Use `compare` method to compare two sequences.
    """

    def compute_auxillary_matrices(self, seqA: str, seqB: str) -> int:
        """
        Computes the score and direction matrices for the SW algorithm.
        Returns the maximum score.

        Args:
            seqA (str): first sequence (horizontal)
            seqB (str): second sequence (vertical)

        Returns:
            int: maximum score
        """
        # prepare a score and direction matrices
        coldim = len(seqA) + 1
        rowdim = len(seqB) + 1
        score_matrix = self._create_initial_score_matrix(rowdim, coldim)
        dir_matrix = self._create_initial_direction_matrix(rowdim, coldim)

        # start computing the squares
        for i in range(1, rowdim):
            for j in range(1, coldim):

                # compute the three possible scores
                score_match = score_matrix[i - 1, j - 1] + self.score(
                    seqB[i - 1], seqA[j - 1]
                )
                score_delete = score_matrix[i - 1, j] + self.GP
                score_insert = score_matrix[i, j - 1] + self.GP

                # choose the maximum NON-NEGATIVE score and update the score matrix
                max_score = max(score_match, score_delete, score_insert, 0)
                score_matrix[i, j] = max(0, max_score)

                # update the direction matrix (multiple directions possible)
                if max_score == score_match:
                    dir_matrix[i][j] += "D"
                if max_score == score_delete:
                    dir_matrix[i][j] += "U"
                if max_score == score_insert:
                    dir_matrix[i][j] += "L"

        # store the matrices
        self.dir_matrix = dir_matrix
        self.score_matrix = score_matrix

        # return the final score
        return np.max(self.score_matrix)

    def _create_initial_score_matrix(self, rowdim: int, coldim: int) -> np.ndarray:
        """
        Creates a score matrix for the SW algorithm.
        The first row and column are initialized with zeros, the rest is to be computed.

        Args:
            rowdim (int): number of rows
            coldim (int): number of columns

        Returns:
            np.ndarray: score matrix
        """
        score_matrix = np.zeros((rowdim, coldim), dtype=int)
        return score_matrix

    def _output_results(self, all_paths, score, output_file=None) -> None:
        """Output the results of local alignments.

        Args:
            output_file (str): path to the output file, if None print to stdout
        """
        output = []
        for i, path in enumerate(all_paths):
            output.append(f"Local alignment no. {i + 1}:")
            for line in path:
                output.append(line)
            output.append(f"Score: {score}\n")
        result = "\n".join(output)
        # write the results to the file or print to stdout
        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(result)
        else:
            print(result)

    def find_paths(self, seqA, seqB, n=1):
        """Uses backtracking via direction matrix to
        find all optimal paths (up to the limit n).
        Starts from the maximum score in the score matrix.
        Ends when a path ends or a zero is reached.

        Args:
            dir_matrix (list[list[str]]): direction matrix
            seqA (str): first sequence (horizontal)
            seqB (str): second sequence (vertical)
            n (int): maximum number of paths to find

        Returns:
            list[tuple[str, str, int]]: list of optimal paths,
            that is tuples (alignementA, alignementB, score)
        """
        # rowdim, coldim = len(self.dir_matrix), len(self.dir_matrix[0])
        global all_paths
        all_paths = []

        # SW Step 0: find the maximum score indices
        max_value = np.max(self.score_matrix)

        def backtrack(i, j, pathA, pathB):
            # early checking condition
            if len(all_paths) >= n:
                return
            # base case: no further path or zero reached
            if "0" in self.dir_matrix[i][j] or self.dir_matrix[i][j] == "":
                all_paths.append((pathA[::-1], pathB[::-1]))
                return
            # recursive cases
            if "D" in self.dir_matrix[i][j]:
                backtrack(i - 1, j - 1, pathA + [seqA[j - 1]], pathB + [seqB[i - 1]])
            if "U" in self.dir_matrix[i][j]:
                backtrack(i - 1, j, pathA + ["-"], pathB + [seqB[i - 1]])
            if "L" in self.dir_matrix[i][j]:
                backtrack(i, j - 1, pathA + [seqA[j - 1]], pathB + ["-"])

        # for each found global maximum score coordinates:
        for x_max, y_max in np.argwhere(self.score_matrix == max_value):
            # trace back the local alignment
            backtrack(x_max, y_max, [], [])

        # change alignments from list of chars to strings
        all_paths = [("".join(pathA), "".join(pathB)) for (pathA, pathB) in all_paths]
        return all_paths


if __name__ == "__main__":
    # TC1 Medium
    # nw = NeedlemanWunsch("data/submatrix3.csv", GP=-5)
    # nw.compare_sequences("CTCGCAGC", "CATTCAC")

    # TC2
    # nw = SmithWaterman("data/submatrix2.csv", GP=-6)
    # nw.compare("TATA", "ATAT", n=2)

    # TC3
    # nw = NeedlemanWunsch("data/submatrix2.csv", GP=-6)
    # nw.compare_sequences("TGCTCGTA", "TTCATA")

    # TC4 Youtube SW
    nw = SmithWaterman("data/submatrix4.csv", GP=-2)
    nw.compare("ATCTG", "ATGTG", n=10)
