import itertools as it

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class CONV:
    def autocorrelation(self, units):
        return np.convolve(units, units)

    def correlation(self, units_1, units_2):
        return np.convolve(units_1, units_2)


class Q1(CONV):
    DT = 0.1
    UNITS = [
        # fmt: off
        0.1, 0.7, 0.8, -0.2, 0.1, 0.1, 0.8, -0.5, 0.2, 0.8, 0.9, 0.1,
        0.2, 0.8, 1.0, -0.3, 0.4, 0.4, 0.8, 0.0, 0.1, 0.2, 0.8, -0.3, -0.4
        # fmt: on
    ]

    def __init__(self) -> None:
        ac = self.autocorrelation(self.UNITS)
        plt.plot(np.arange(0, self.DT * len(ac), self.DT), ac)
        plt.show()
        # ulangan sekitar setiap 2 saat


class Q2(CONV):
    DT = 0.1
    UNITS_1 = [0.0, -0.1, -0.3, 0.1, -0.3, 0.7, 0.6, -0.1, -0.1, 0.2, -0.1, -0.3]
    UNITS_2 = [0.0, 0.6, 1.2, -0.3]

    def __init__(self) -> None:
        ac = self.correlation(self.UNITS_1, self.UNITS_2)
        plt.plot(np.arange(0, self.DT * len(ac), self.DT), ac)
        plt.show()
        # tiba sekitar 0.5 saat


class Q3:
    PLOT_KW = {
        "cmap": "gray",
        "aspect": "auto",
        "interpolation": "nearest",
    }

    def __init__(self) -> None:
        image = np.array(Image.open("./assets/image002.jpg"))

        # constrasting
        CONSTRAST_THRES = 0.5
        image = (image / np.max(image) > CONSTRAST_THRES).astype("u8")

        plt.subplot(2, 2, 1)
        plt.xlabel("Contrasted Original")
        plt.imshow(image, **self.PLOT_KW)

        LINES_THRES = 0.6
        arr, R_max, T_max = self.hugh(image)
        plt.subplot(2, 2, 2)
        plt.xlabel("Hugh Transform")
        plt.imshow(arr, extent=[0, T_max, 0, R_max], **self.PLOT_KW)

        arr = (arr / np.max(arr) > LINES_THRES).astype("u8")
        plt.subplot(2, 2, 3)
        plt.xlabel("Thresholded bins\npossible amount of lines")
        plt.imshow(arr, extent=[0, T_max, 0, R_max], **self.PLOT_KW)
        print(arr.sum())
        plt.show()

    def hugh(self, image: np.ndarray):
        (I, J), MAX = image.shape, np.max(image)
        R_max, TD_max = np.linalg.norm([I, J]), 180

        # samples for both distance and angle
        N, M, T_MAX = round(R_max), round(TD_max), np.deg2rad(TD_max)
        arr = np.zeros((N, M))

        P = np.array(np.where(image == MAX)).T
        T = np.linspace(0, T_MAX, M)
        for p, t in it.product(P, T):
            r = self.get_shortest_dist(p, t)
            r_ind = round(r / R_max * (N - 1))
            t_ind = round(t / T_MAX * (M - 1))
            arr[r_ind, t_ind] += MAX

        return arr, R_max, TD_max

    def get_shortest_dist(self, p: np.ndarray, theta: float):
        "2D distance"
        return np.linalg.norm(np.cross(p, [-np.sin(theta), np.cos(theta)]))


class Q4:
    T = 10
    g, dt = 9.81, 1e-2
    ts = np.arange(0, T, dt)
    np.random.seed(1)

    # prediction parameters
    F = np.array([[1, dt], [0, 1]])
    G_u = np.array([[-0.5 * dt**2], [-dt]]) * g
    Q = np.zeros((2, 2))

    # measurement parameters
    H = np.array([[1, 0]])
    sig = 0.1

    def __init__(self) -> None:
        XS, ZS = [], []
        X, Z = np.array([[105], [0]]), np.array([[100], [0]])
        P = np.array([[10, 0], [0, 0.01]])

        for _ in self.ts:
            Z = self.F @ Z + self.G_u
            X, P = self.get_prediction(X, P)
            X, P = self.update(X, P, Z)
            XS += [X]
            ZS += [Z]

        XS, ZS = np.array(XS), np.array(ZS)
        plt.plot(self.ts, XS[:, 0], label="prediction")
        plt.plot(self.ts, ZS[:, 0], label="true")
        plt.legend()
        plt.show()

    def get_prediction(self, X, P):
        X = self.F @ X + self.G_u + np.random.randint(-1000, 1000) / 10
        P = self.F @ P @ self.F.T + self.Q
        return X, P

    def update(self, X, P, Z):
        R = np.random.randint(-1000, 1000) / 100
        K = P @ self.H.T @ np.linalg.inv(self.H @ P @ self.H.T + R)
        X = X + K * self.H @ (Z - X)
        P = (1 - K @ self.H) @ P
        return X, P


Q1()
Q2()
Q3()
Q4()
