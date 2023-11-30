import numpy as np
from numba import jit


class Q1:
    def main(self):
        # _2x2 = np.arange(1, 5).reshape(2, 2)
        # print("det of ", _2x2, " is ", self.two_det(_2x2))
        # _3x3 = np.arange(1, 10).reshape(3, 3)
        # print("det of ", _3x3, " is ", self.three_det(_3x3))
        _10x10 = np.arange(1, 101).reshape(10, 10)
        print("det of ", _10x10, " is ", self.det(_10x10))

    def make_upper_diag(self, A):
        n = len(A)
        np.meshgrid([range(n)])

    def one_det(self, A):
        return np.array(A)[0, 0]

    def submatrix(self, A, i, n):
        i, j = np.meshgrid(
            [j for j in range(n + 1) if j != i],
            [j for j in range(1, n + 1)],
        )
        return np.array(A)[j, i]

    def two_det(self, A):
        D = 0
        for n, a in enumerate(A[0]):
            D += (
                (-1) ** n
                * a
                * self.one_det(
                    self.submatrix(A, n, 1),
                )
            )
        return D

    def three_det(self, A):
        D = 0
        for n, a in enumerate(A[0]):
            D += (
                (-1) ** n
                * a
                * self.two_det(
                    self.submatrix(A, n, 2),
                )
            )
        return D

    @jit
    def det(self, A):
        D = 0
        if len(A) == 1:
            return self.one_det(A)
        if len(A) == 2:
            return self.two_det(A)
        if len(A) == 3:
            return self.three_det(A)

        for n, a in enumerate(A[0]):
            D += (
                (-1) ** n
                * a
                * self.det(
                    self.submatrix(A, n, len(A) - 1),
                )
            )
        return D


if __name__ == "__main__":
    Q1().main()
