import numpy as np
import matplotlib.pyplot as plt


class Q1:
    L = 4.0
    h = 1.0
    T = 10  # seconds in simulation
    dt = 1e-2

    def __init__(self):
        xs, ys = 0, 0
        # for i in [1, 2, 3, 4, 8, 16]:
        #     xs, ys = self.fourier_form(i)
        #     plt.show()
        xs, ys = self.fourier_form(16)
        plt.clf()
        self.plot_wave(xs, ys, 16)
        plt.show()

    def fourier_form(self, J=16):
        self.J = J
        j = np.arange(J)
        xs = np.linspace(0, self.L, 100)
        ys1 = self.f(xs)
        plt.title(f"J = {J}")
        plt.plot(xs, ys1)

        ys2 = np.sum(
            [self.a(k, xs, ys1) * np.sin(k * np.pi * xs / self.L) for k in j],
            axis=0,
        )
        plt.plot(xs, ys2)

        return xs, ys2

    def a(self, k, xs, ys):
        dx = xs[1] - xs[0]
        return 2 / self.L * np.sum([ys * np.sin(k * np.pi * xs / self.L) * dx])

    def plot_wave(self, xs, fx0, J):
        self.J = J
        j = np.arange(J)
        T = int(self.T / self.dt)
        ft = np.zeros((T, len(xs)))
        ft_t = np.zeros((T, len(xs)))
        ft[0] = fx0
        for t in range(1, T):
            fx_xx = -np.sum(
                [
                    self.a(k, xs, ft[t - 1])
                    * (k * np.pi / self.L) ** 2
                    * np.sin(k * np.pi * xs / self.L)
                    for k in j
                ],
                axis=0,
            )
            # kaedah Euler
            ft_t[t] = ft_t[t - 1] + fx_xx * self.dt
            ft[t] = ft[t - 1] + ft_t[t] * self.dt

        ax = plt.axes(projection="3d")
        x, y = np.meshgrid(xs, np.arange(T) * self.dt)
        ax.plot_surface(x, y, ft)
        ax.set_title("Bentuk Tangsi Terhadap Masa")
        ax.set_xlabel("Ruang")
        ax.set_ylabel("Masa")
        ax.set_zlabel("Tinggi")

    def f(self, xs):
        return np.vectorize(
            lambda x: (2 * self.h / self.L) * x
            if x < self.L / 2
            else 2 - (2 * self.h / self.L) * x
        )(xs)


class Q2:
    hbar, mass = 1, 1
    a = -1j * hbar / 2 / mass
    b = 0
    c = -1j * hbar  # to be multiplied with V
    h, k = 0.5, 0.1
    T = 20  # seconds of simulation

    def __init__(self):
        xs = np.arange(-20, 20 + self.h, self.h)
        ts = int(self.T / self.k)
        us = np.zeros((ts, len(xs)), complex)
        us[0] = np.vectorize(self.f0)(xs)
        us[0] /= np.trapz(abs(us[0]) ** 2, xs) ** 0.5

        for t in range(1, ts):
            usp = us[t - 1]
            us[t] = (np.linalg.inv(self.CN_mat(xs)) @ [usp[0], *usp, usp[-1]])[1:-1]
            us[t] /= np.trapz(abs(us[t]) ** 2, xs) ** 0.5

        ax = plt.axes(projection="3d")
        ax.set_title("Kebarangkalian")
        ax.set_xlabel("Ruang")
        ax.set_ylabel("Masa")
        x, y = np.meshgrid(xs, np.arange(ts) * self.k)
        ax.plot_surface(x, y, abs(us) ** 2)
        plt.show()

    def f0(self, x):
        # Fungsi gelombang dengan nilai momentum, k=4
        return (10) ** 0.5 / 20 * np.exp(-(x + 5) * (x + 5 - 320j) / 80)

    def CN_mat(self, xs):
        xs = [xs[0], *xs, xs[-1]]
        mat = np.zeros((len(xs), len(xs)), complex)
        tridi00 = np.array(np.diag_indices_from(mat))
        tridi_1 = (tridi00.T + [1, 0]).T[:, :-1]
        tridi01 = (tridi00.T + [0, 1]).T[:, :-1]
        mat[tridi00[0], tridi00[1]] = np.vectorize(self.B)(xs)
        mat[tridi_1[0], tridi_1[1]] = np.vectorize(self.A)(xs[1:])
        mat[tridi01[0], tridi01[1]] = np.vectorize(self.C)(xs[:-1])
        return mat

    def A(self, _):
        return 2 * self.k * self.a + self.k * self.h * self.b

    def B(self, x):
        return (
            4 * self.h**2
            - 4 * self.k * self.a
            + 2 * self.h**2 * self.k * self.c * self.V(x)
        )

    def C(self, _):
        return 2 * self.k * self.a - self.k * self.h * self.b

    def D(self, u3):
        return np.sum(self.coeffs(u3[1]) * u3)

    def coeffs(self, x):
        return np.array(
            [self.A(x), self.B(x), self.C(x)],
            complex,
        )

    def V(self, x):
        # return 0
        return 0 if abs(x - 50) < 3 else 100


if __name__ == "__main__":
    Q1()
    Q2()
