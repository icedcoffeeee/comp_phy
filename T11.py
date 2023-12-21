import numpy as np
import itertools as it
import matplotlib.pyplot as plt

np.random.seed(1)


class Q1:
    J, B = 1, 1
    beta = 1

    def __init__(self) -> None:
        for N in (10, 100):
            self.sim(N)
        # berlaku perubahan fasa bagi nilai N yg meningkat

    def sim(self, N=10) -> None:
        S = np.random.choice([-1, 1], (N, N), p=[0.5, 0.5])

        ts = range(10)
        Hs, Ms = [], []
        for _ in ts:
            S = self.metropolis(S)
            Hs += [self.get_H(S).sum()]
            Ms += [S.sum() / N**2]

        plt.plot(ts, Hs, ts, Ms)
        plt.show()

    def metropolis(self, S):
        return S * np.vectorize(
            lambda p: np.random.choice(
                [-1, 1],
                p=[min(p, 1), max(1 - p, 0)],
            )
        )(self.get_P(S))

    def get_P(self, S):
        return np.exp(-self.beta * self.get_H(S))

    def get_H(self, S):
        return -self.J * self.get_S_surr(S) - self.B * S.sum()

    def get_S_surr(self, S):
        I, J = S.shape
        pad = np.pad(S, 1, mode="edge")
        repeats = np.repeat(S, 8, -1).reshape(I, J, 8)
        stacked = np.transpose(
            [
                np.roll(pad, 0, 0),
                np.roll(pad, 1, 0),
                np.roll(pad, 2, 0),
                np.roll(pad, 1, 1),
                np.roll(np.roll(pad, 1, 1), 2, 0),
                np.roll(pad, 2, 1),
                np.roll(np.roll(pad, 2, 1), 1, 0),
                np.roll(np.roll(pad, 2, 1), 2, 0),
            ],
        )[1:-1, 1:-1]
        S_surr = np.prod([repeats, stacked], axis=0).sum(axis=-1)

        return S_surr


class Q2:
    N = 100
    Tlims = [-2, 2]

    def __init__(self) -> None:
        self.init_weights()

        ts = np.arange(0, 10, 0.5)
        Es = []
        for t in ts:
            self.V = self.updated_V()
            print("t=", t, ":", self.V)
            Es += [self.get_E()]

        plt.title("Energy over Time")
        plt.plot(ts, Es)
        plt.show()

    def init_weights(self):
        self.V = np.random.random(self.N).round().astype(int)
        self.U = np.zeros(self.N)

        T = np.random.random((self.N, self.N))
        A, B = self.Tlims
        self.T = A + T * (B - A)  # normalize between limits

        self.T[range(self.N), range(self.N)] = 0  # T_ii = 0
        L = np.tril_indices_from(self.T)
        self.T[L] = self.T.T[L]  # T_ij = T_ji

    def get_E(self):
        return -0.5 * self.T @ self.V @ self.V + self.U @ self.V

    def get_dEdV(self):
        return -self.T @ self.V + self.U

    def updated_V(self):
        return self.heaviside(-self.get_dEdV())

    def heaviside(self, t):
        return np.int32(t > 0)


Q, W, O = np.zeros((3, 100))
# fmt: off
Q[[
         4,  5,  6,
    13,         16,
    23,         26,
        34, 35, 36,
                46,
                56,
                66,
                76,     78,
                86, 87,
                96
]] = 1
W[[
    20,         23,                     29,
    30,         33,                     39,
        41,     43, 44,             48,
            52, 53,     55,     57,
                65,         66,
]] = 1
O[[
        13, 14, 15, 16,
    22,                 27,
    32,                 37,
    42,                 47,
    52,                 57,
    62,                 67,
        73, 74, 75, 76,
]] = 1
# fmt: on


class Q3(Q2):
    N = 100

    def __init__(self) -> None:
        self.xis = Q, W, O
        self.init_weights()

        ts = np.arange(0, 50, 0.01)
        Es = []
        for _ in ts:
            self.V = self.updated_V()
            Es += [self.get_E()]

        M = int(self.N**0.5)
        plt.title("Nilai-nilai T")
        plt.pcolormesh(*np.meshgrid(*[np.arange(M)] * 2), self.V.reshape(M, M)[::-1])
        plt.show()

    def init_weights(self):
        N = self.N
        self.V = np.random.random(N)
        self.U = np.zeros(N)
        self.T = np.sum([np.outer(xi, xi) for xi in self.xis], 0)


class Q4:
    N = [100, 10, 3]
    eta = 1
    tol = 0.05

    def __init__(self) -> None:
        self.init_weights()
        for _ in range(150):
            self.train(Q, [1, 0, 0])
            self.train(W, [0, 1, 0])
            self.train(O, [0, 0, 1])

        # testing memory
        print(self.forward(Q).round())
        print(self.forward(W).round())
        print(self.forward(O).round())

        # testing generalization
        print(self.forward(np.random.random(100)).round())

    def init_weights(self):
        N = self.N
        T = []
        for s in zip(N, N[1:]):
            T += [2 * np.random.random(s[::-1]) - 1]
        self.T = T

    def train(self, i, o):
        for T in self.T[::-1]:
            Y, X = T.shape
            for y, x in it.product(range(Y), range(X)):
                e0 = self.get_error(i, o)
                T[y, x] += self.eta
                e1 = self.get_error(i, o)
                if abs(e1 - e0) < self.tol:
                    T[y, x] -= self.eta
                elif e1 - e0 > 0:
                    T[y, x] -= 2 * self.eta

    def get_error(self, i, o):
        v = self.forward(i)
        return np.sum((o - v) ** 2)

    def forward(self, i):
        v = i
        for T in self.T:
            v = T @ v
        return self.activation(v)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))


Q1()
Q2()
Q3()
Q4()
