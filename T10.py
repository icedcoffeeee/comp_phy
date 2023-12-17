from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Q1:
    np.random.seed(1)

    def __init__(self) -> None:
        self.N = int(1e5)
        n = np.arange(self.N)

        LR = np.random.choice([-0.1, 0.1], self.N - 1, [0.5, 0.5])
        ys = np.append([0], np.cumsum(LR))
        vs = [np.mean(ys[: i + 1] ** 2) for i in n]

        plt.plot(n, ys, n, n**0.5, n, vs)
        plt.legend(["Distance", r"$\sqrt{n}$", "RMS value"])
        plt.xlabel("Number of steps, $n$")

        plt.show()


class Q2:
    np.random.seed(1)
    N = 100

    def __init__(self) -> None:
        groups = self.get_groups_from_P(0.3)
        print(sum([len(i) for i in groups]) / (len(groups) or 1))

        ps = np.linspace(0, 1, 10)
        ls = [len(self.get_groups_from_P(i)) for i in ps]
        plt.plot(ps, ls)
        plt.show()

    def get_groups_from_P(self, P: float) -> None:
        cells = np.random.choice([0, 1], (self.N, self.N), p=[1 - P, P])
        positions = np.indices(cells.shape).T[cells == 1].reshape(-1, 2).tolist()
        groups = self.group_positions(positions)
        print(len(groups))
        return groups

    def group_positions(self, positions: Iterable[Tuple[int, int]]):
        groups = []
        for m, (x, y) in enumerate(positions):
            N = -1
            for n, g in enumerate(groups):
                for gp in g:
                    if [x, y] == gp:  # get point's group index
                        N = n

            if N == -1:  # point not yet in a group
                groups += [[[x, y]]]

            M = 0
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                neighbour = [x + dx, y + dy]
                if neighbour in positions:
                    M += 1
                    if neighbour not in groups[N]:
                        groups[N] += [neighbour]
                        positions.remove(neighbour)
                        positions.insert(m + M, neighbour)
        return groups


class Q3:
    np.random.seed(1)

    def __init__(self) -> None:
        n = 100
        self.N = 10
        self.MAX = 1
        self.space = range(self.N)
        self.grains = np.zeros(self.N).astype(int)
        self.falls = 0
        [self.add_grain(pos) for pos in np.random.uniform(0, self.N, n).astype(int)]

        plt.title("grains against bins")
        plt.xlabel("bin")
        plt.ylabel("grains")
        plt.plot(self.space, self.grains)
        plt.show()

        num_falls = []
        for n in range(101):
            self.falls = 0
            [self.add_grain(pos) for pos in np.random.uniform(0, self.N, n).astype(int)]
            num_falls += [self.falls]

        plt.title("collapses against sand drops")
        plt.xlabel("sand drops")
        plt.ylabel("collapses")
        plt.plot(range(101), num_falls)
        plt.show()


    def add_grain(self, n: int):
        self.falls += 1

        if n <= 0 or n >= self.N - 1:
            return

        match self.grains[n]:
            case 0:
                self.grains[n] += 1
            case _:
                self.grains[n] -= 1
                self.add_grain(n + 1)
                self.add_grain(n - 1)


Q1()
Q2()
Q3()
