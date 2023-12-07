import random
import time
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


class Q1_2:
    def __init__(self, Q=1):
        self.N = 100
        values = []

        match Q:
            case 1:
                values = self.get_sequence(self.get_rand_1, 0)
            case 2:
                J = 239475
                values = self.get_sequence(self.get_rand_2, J)
                values = [i**0.5 / 1000 for i in values]

        self.plot_sequence(values)
        self.plot_correlation(values)

    def get_sequence(self, f: Callable, *args):
        rand = f(*args)
        values = [rand]
        for _ in range(self.N - 1):
            values += [f(values[-1])]
        return values

    def get_rand_1(self, after: float):
        time.sleep(after / 1000)
        return float(str(time.time())[-3:]) / 1000

    def get_rand_2(self, J: int):
        smallest_3 = int("".join(sorted([*str(J)])[:3]))
        return smallest_3**2

    def plot_sequence(self, vals: Iterable[float]):
        plt.plot(range(len(vals)), vals)
        plt.show()

    def plot_correlation(self, vals: Iterable[float]):
        u, v = vals[:-1], vals[1:]
        plt.scatter(u, v)
        plt.show()


class Q3:
    def __init__(self) -> None:
        N = 0
        vals = []
        while len(vals) < 1000:
            point = x, _ = [random.random() * 2, random.random()]
            if point < [x, self.func(x)]:
                vals += [point]
            N += 1
        vals = [x for x, _ in vals]

        plt.hist(vals, 100)
        plt.show()

        print("Inside:", len(vals), "\tTotal:", N)
        print("Ratio:", len(vals) / N)
        print("Area:", len(vals) / N * 1.0 * 2.0)

    def func(self, x):
        return x if x < 1 else 2 - x


class Q4:
    def __init__(self) -> None:
        self.M = np.pi / 3  # sudut medan magnet
        self.angleN = 100
        np.random.seed(100)
        A = 5000  # mula dengan sebilangan A
        decay_types = self.get_decay_types(A)
        decay_angles = self.get_decay_angle(decay_types)

        B_ang = decay_angles[decay_types == 1]
        C_ang = decay_angles[decay_types == 1]
        D_ang = decay_angles[decay_types == 2]
        By, Bx = np.histogram(B_ang, int(len(B_ang) / 10))
        Cy, Cx = np.histogram(C_ang, int(len(C_ang) / 10))
        Dy, Dx = np.histogram(D_ang, int(len(D_ang) / 10))

        Bx, By = Bx[1:][By != 0], -By[By != 0]  # keabadian momentum
        Cx, Cy = Cx[1:][Cy != 0], Cy[Cy != 0]
        Dx, Dy = Dx[1:][Dy != 0], Dy[Dy != 0]

        plt.plot(Bx, By, lw=1, label="B")
        plt.plot(Cx, Cy, lw=1, label="C")
        plt.plot(Dx, Dy, lw=1, label="D")

        plt.title("Momentum lawan Sudut")
        plt.xlabel("Sudut")
        plt.ylabel("Momentum")
        plt.legend()
        plt.show()

    def get_decay_types(self, num: int):
        return np.random.choice([1, 2], num, p=[0.7, 0.3])

    def get_decay_angle(self, types: Iterable[int]):
        self.angle_range = np.linspace(0, 2 * np.pi, self.angleN)
        norm = lambda v: v / sum(v)
        self.angle_probs = norm(self.C_distr(self.angle_range))
        return np.vectorize(self.get_angle_from_type)(types)

    def get_angle_from_type(self, type: int):
        match type:
            case 1:
                # pulangkan sudut C
                # B hanya berlawanan
                return np.random.choice(self.angle_range, p=self.angle_probs)
            case 2:
                return np.random.uniform(0, 2 * np.pi)

    def C_distr(self, th):
        return np.exp(-abs(th - self.M))


# Q1_2(1)
# Q1_2(2)
# Q3()
Q4()
