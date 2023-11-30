import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess


class Q1:
    dt, T = 0.1, 1
    tol = 1e-2

    du = lambda _, v: v
    dv = lambda _, u: -np.pi**2 / 4 * (u + 1)

    np.random.seed(100)
    t = np.arange(0, T, dt)

    def main(self):
        # two random guesses
        v1, v2 = np.random.uniform(-3, 3, (2,))
        pv1, pv2 = self.propagate(v1), self.propagate(v2)
        while pv1 * pv2 > 0:
            v1, v2 = np.random.uniform(-3, 3, (2,))
            pv1, pv2 = self.propagate(v1), self.propagate(v2)

        v1 = self.secant(v1, v2, self.propagate)
        print(v1, self.propagate(v1))

    def phaseplot(self):
        xs, ys = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
        us, vs = self.du(ys), self.dv(xs)
        plt.quiver(xs, ys, us, vs)
        plt.show()

    def RK(self, y, f):
        h = self.dt
        K1 = f(y)
        K2 = f(y + K1 * h / 2)
        K3 = f(y + K2 * h / 2)
        K4 = f(y + K3 * h)
        return y + (K1 + 2 * K2 + 2 * K3 + K4) * h / 6

    def propagate(self, v):
        u = 0  # u(0) = 0

        for _ in self.t:
            v += self.RK(u, self.dv)
            u += self.RK(v, self.du)

        return u - 1  # cost (to obtain the root)

    def secant(self, a, b, f):
        if abs(f(a)) < self.tol:
            return a
        c = a - f(a) * (a - b) / (f(a) - f(b))
        if f(c) * f(a) > 0:
            return self.secant(c, b, f)
        return self.secant(a, c, f)


class Q2:
    """
    N particles in a L length 1-D box.
    """

    L = 100
    N = 10_000
    T = 2
    velocities = np.zeros(N)

    q, m, dt = 1, 1, 1 / 60
    EPS0 = 1  # permittivity
    COND = 1  # conductivity
    np.random.seed(100)

    deriv = np.zeros((L, L))
    deriv[range(L - 1), range(1, L)] = 1
    deriv[range(1, L), range(L - 1)] = -1
    # deriv[[0, -1], [-1, 0]] = [-1, 1]

    deriv2 = np.zeros((L, L))
    deriv2[range(L), range(L)] = -2
    deriv2[range(L - 1), range(1, L)] = 1
    deriv2[range(1, L), range(L - 1)] = 1
    integ2 = np.linalg.inv(deriv2)

    def main(self, show=1):
        data = [("P", "f8"), ("C", "f8"), ("V", "f8"), ("E", "f8")]
        self.data = np.empty((self.L,), dtype=data)
        self.data["P"] = np.linspace(0, 1, self.L)

        func = np.vectorize(lambda x: x**2 * np.cos(np.pi * x / 2))
        # func = np.vectorize(lambda x: 1.0)
        # func = np.vectorize(lambda x: (x - 0.5) ** 2)
        self.data["C"] = func(self.data["P"])
        # edge cases
        self.data["C"][[0, -1]] = 0
        # normalization
        self.data["C"] /= np.sum(self.data["C"]) / self.N
        # average for redistribution
        # avg = np.average(self.data["C"] * self.P)

        self.data["V"] = -self.int2_dx(self.data["C"])
        self.data["E"] = -self.deriv_dx(self.data["V"])

        fig = plt.figure()
        ax = fig.add_subplot()
        plots = ax.plot(
            # sloppy data
            self.data["P"],
            self.data["C"],
            self.data["P"],
            self.data["V"],
            self.data["P"],
            self.data["E"],
        )
        ax.legend(["charge", "potential", "efield"])

        def animate(i):
            if i % 2 == 0:
                # every other step, update potential
                self.data["V"] = -self.int2_dx(self.data["C"])
                self.data["E"] = -self.deriv_dx(self.data["V"])
            else:
                # else, update charge distribution
                self.data["C"] -= (
                    self.deriv_dx(self.data["E"])
                    * self.dt
                    # - self.N / self.L * self.dt
                    # ^^ redistribution term
                )
                # edge case
                self.data["C"][[0, -1]] = 0
                # normalization
                # self.data["C"] /= np.sum(self.data["C"]) / self.N

            for p, (y, _) in zip(plots, data[1:]):
                p.set_data(self.data["P"], self.data[y])

            return plots

        if not show:
            anim = animation.FuncAnimation(
                fig,
                animate,
                int(self.T / self.dt),
                interval=int(self.dt * 1000),
                blit=True,
            )
            anim.save("A3_Q2.mp4")
            subprocess.Popen("A3_Q2.mp4", shell=1)
        else:
            plt.show()

    def int2_dx(self, integrand):
        return self.integ2 @ integrand / self.L**2

    def deriv_dx(self, derivand):
        return self.deriv @ derivand * self.L / 2


if __name__ == "__main__":
    # Q1().main()
    Q2().main(show=0)
