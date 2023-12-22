from typing import Iterable
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

np.random.seed(1)


class Q1:
    mu = 6 + 2.5

    def __init__(self) -> None:
        poisson = [*self.get_poisson(self.mu)]
        n_max = len(poisson)
        n_random = self.monte_carlo(1000, n_max, poisson)

        plt.hist(n_random, n_max - 1)
        plt.show()

        grouped = self.group_by_neighbours(n_random)
        grouped[grouped == 0] = 1e-7
        plt.scatter(range(len(grouped)), 1 / grouped)
        plt.show()

        print(np.mean(n_random), np.var(n_random))
        # nilai min dan variansnya menghampiri nilai mu

    def get_poisson(self, mu: float):
        P = 1
        n = -1
        cut = False
        while not cut or P > 0.01:
            n += 1
            P = np.exp(-mu) * mu**n / self.fact(n)
            if P > 0.01:
                cut = True
            yield P

    def fact(self, n):
        if n in [0, 1]:
            return 1
        elif n > 1:
            return n * self.fact(n - 1)
        raise ValueError()

    def monte_carlo(self, N: int, n_max: int, poisson: Iterable[float]):
        amounts = np.zeros(n_max)
        rands = []
        poisson = (poisson / np.sum(poisson) * 1000).round()

        n = 1
        while n <= N:
            randint = np.random.randint(0, n_max)
            if amounts[randint] < poisson[randint]:
                amounts[randint] += 1
                rands += [randint]
                n += 1

        return rands

    def group_by_neighbours(self, vals: Iterable[int]):
        closest = []
        for v1 in vals:
            dist = 100
            for v2 in vals:
                if abs(v2 - v1) < dist and v2 != v1:
                    dist = v2 - v1
            closest += [dist]
        return np.array(closest)


class Q2:
    VALS = [
        # fmt: off
        7.117, 4.689, 1.718, 8.062, 3.117, 3.994, 3.821, 1.981, 3.730, 2.817,
		4.204, 1.242, 5.017, 3.501, 2.518, 2.692, 3.055, 2.669, 5.196, 2.463,
		0.252, 4.779, 3.545, 3.712, 2.908, 3.553, 1.485, 2.301, 2.937, 5.789,
		1.108, 1.657, 2.624, 1.697, 3.504, 6.052, 2.383, 3.923, 4.257, 5.337,
		5.264, 3.910, 2.783, 2.727, 1.427, 5.250, 4.315, 3.023, 3.556, 2.964,
		2.516, 3.108, 3.898, 2.609, 4.216, 3.066, 3.457, 5.214, 4.302, 5.458,
		4.628, 3.519, 3.736, 4.986, 1.444, 5.675, 4.146, 1.790, 3.111, 2.544,
		0.272, 4.438, 2.673, 2.321, 2.698, 3.504, 3.738, 3.077, 2.880, 3.339,
		2.602, 1.980, 3.247, 3.165, 3.721, 2.616, 3.535, 2.519, 2.950, 3.648,
		1.400, 1.903, 1.698, 5.148, 4.023, 1.738, 3.841, 3.588, 3.379, 4.131,
		3.776, 1.953, 1.293, 2.184, 2.840, 1.431, 1.009, 1.621, 3.868, 3.209,
		2.864, 2.552, 3.378, 2.777, 3.002, 3.135, 2.588, 9.685, 3.117, 2.824,
		0.497, 3.002, 2.559, 3.490, 3.574, 5.457, 5.040, 2.946, 3.417, 3.302,
		1.287, 2.644, 6.086, 3.102, 2.414, 5.277, 2.307, 4.122, 2.727, 2.948,
		3.160, 1.937, 3.634, 5.564, 2.865, 3.424, 5.521, 3.102, 3.130, 2.755,
		3.342, 4.907, 5.577, 1.925, 1.754, 4.208, 2.047, 1.348, 3.768, 3.668,
		3.142, 2.911, 1.624, 3.897, 2.918, 6.032, 3.750, 7.921, 2.717, 1.127,
		5.814, 1.662, 3.902, 2.667, 3.276, 2.345, 4.925, 2.487, 3.436, 2.239,
		2.509, 3.437, 2.476, 5.606, 0.993, 4.471, 2.212, 3.100, 3.737, 2.474,
		2.905, 3.713, 4.550, 3.725, 6.697, 4.383, 2.470, 6.189, 2.993, 3.369,
        # fmt: on
    ]

    def __init__(self) -> None:
        plt.hist(self.VALS)

        amounts, bins = np.histogram(self.VALS)
        midpoints = np.array(bins[1:]) - (bins[1] - bins[0]) / 2
        errors = amounts**0.5
        plt.errorbar(midpoints, amounts, errors, fmt="o", capsize=2)

        (x0, gamma), _ = opt.curve_fit(self.BW, midpoints, amounts)
        plt.plot(midpoints, self.BW(midpoints, x0, gamma))

        plt.show()

    def BW(self, x, x0, gamma):
        "Breit-Wigner"
        return 1 / ((x - x0) ** 2 + gamma**2)


Q1()
Q2()
