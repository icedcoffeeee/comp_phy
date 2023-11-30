import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative as dv


class Q1:
    hbar, m = 1, 1
    A, L = 1, 4
    dx = 0.2
    xs = np.arange(-8, 8 + dx, dx)
    tol = 0.5

    def __init__(self):
        # E0 = self.newton(0, self.F)
        E0 = self.secant(-1.9, -1.2, self.F)
        print(E0)  # -1.6045470272363627

        # penyelesaian analisisan memberi nilai -1.5,
        # ralat 7%
        self.plot_V()
        self.LR_ys(E0, 1)
        plt.legend()
        plt.show()

    def F(self, E, plot: bool = False):
        ys_left, ys_right = self.LR_ys(E, plot)
        dleft = (ys_left[-1] - ys_left[-2]) / self.dx
        dright = -(ys_right[-1] - ys_right[-2]) / self.dx
        # print(E, dleft, dright)
        return (dleft - dright) / (2 * self.dx)

    def LR_ys(self, E, plot=False):
        xs = self.xs
        ys_left, x1 = self.ys(E, lambda x: np.isclose(self.V(x), E, atol=self.tol))
        ys_right, x2 = self.ys(E, lambda x: x >= -x1)
        ys_left, ys_right = np.array(ys_left), np.array(ys_right)
        ys_right *= ys_left[-1] / ys_right[-1]
        if plot:  # for finalization
            A = np.trapz(np.append(ys_left, ys_right[:-1]), xs)
            plt.plot(xs[xs <= x1], ys_left / A, label="left")
            plt.plot(-xs[xs <= x2], ys_right / A, label="right")
        return ys_left, ys_right

    def numerov_next(self, xn, xnm1, yn, ynm1, E):
        h12 = self.dx**2 / 12
        gn, gnm1, gnp1 = self.G(xn, E), self.G(xnm1, E), self.G(xn + self.dx, E)
        return (2 * yn * (1 - 5 * h12 * gn) - ynm1 * (1 + h12 * gnm1)) / (
            1 + h12 * gnp1
        )

    def G(self, x, E):
        return 2 * self.m / self.hbar**2 * (E - self.V(x))

    def V(self, x):
        return (
            self.hbar**2
            / (2 * self.m)
            * self.A**2
            * self.L
            * (self.L - 1)
            * (0.5 - np.cosh(self.A * x) ** -2)
        )

    def plot_V(self):
        return plt.plot(self.xs, np.vectorize(self.V)(self.xs), label="potential")

    def ys(self, E, cond, out=False):
        ys = [1e-3, 1e-3]
        x = 0
        for x in self.xs[2:]:
            ys.append(self.numerov_next(x, x - self.dx, ys[-1], ys[-2], E))
            if out:
                print(ys[-1])
            if cond(x):
                break
        return ys, x

    def secant(self, a, b, f):
        if f(a) * f(b) > 0:
            raise ValueError("root not between a and b")
        if abs(f(a)) < self.tol:
            return a
        c = a - f(a) * (a - b) / (f(a) - f(b))
        if f(c) * f(a) > 0:
            return self.secant(c, b, f)
        return self.secant(a, c, f)

    def newton(self, x0, f):
        # print(x0)
        if abs(f(x0)) < self.tol:
            return x0
        return self.newton(x0 - f(x0) / dv(f, x0, self.dx), f)


if __name__ == "__main__":
    Q1()
