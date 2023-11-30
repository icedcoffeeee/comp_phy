import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def soalan1():
    # medan elektrik malar ruang dan masa
    e, m = 1, 1  # cas dan jisim
    E = lambda r, t: r / np.linalg.norm(r) ** 3
    a = lambda x, t: e / m * E(x, t)  # Pecutan

    # parameter simulasi
    dt = 0.1
    ts = np.arange(0, 10, dt)  # simulasi sehingga 10 saat

    # nilai permulaan
    x0, v0 = np.array([-10, 2, 0]), np.array([1, 0, 0])
    xs, vs, Fs, Us = [x0], [v0 + a(x0, ts[0]) * dt], [], [0]

    for t in ts[1:]:
        xs.append(xs[-1] + 2 * vs[-1] * dt)
        vs.append(vs[-1] + 2 * a(xs[-1], t) * dt)
        Fs.append(e * E(xs[-1], t))

    xs = np.array(xs)
    vs = np.array(vs)
    Fs = np.array(Fs)
    for F, x0, x1 in zip(Fs, xs, xs[1:]):
        Us.append(Us[-1] - F @ (x1 - x0))

    plt.plot([0], [0], "ro", label="Cas titik")
    plt.plot(xs[:, 0], xs[:, 1], label="Kedudukan, $r(t)$")
    plt.quiver(xs[::5, 0], xs[::5, 1], vs[::5, 0], vs[::5, 1], label="Halaju, $v(t)$")
    plt.title("Gerakan Zarah")
    plt.ylabel("$y$")
    plt.xlabel("$x$")
    plt.legend()
    plt.savefig("A2_1_1.png")
    plt.show()
    plt.plot(ts, Us, label="Tenaga Keupayaan")
    plt.plot(
        ts + dt / 2,
        m / 2 * np.linalg.norm(vs, axis=1) ** 2,
        label="Tenaga Kinetik",
    )
    plt.title("Tenaga Zarah")
    plt.xlabel("Masa, t")
    plt.legend()
    plt.savefig("A2_1_2.png")
    plt.show()


def soalan2():
    e, m = 0.1, 1  # cas dan jisim
    # parameter simulasi
    dt = 0.1
    ts = np.arange(0, 10, dt)  # simulasi sehingga 10 saat

    def get_acc(pos):
        "Fungsi Pecutan"
        r_diffs = np.array([i - j for i in pos for j in pos])
        r_diffs = r_diffs.reshape(len(pos), len(pos), 2)
        distances = np.linalg.norm(r_diffs, axis=2)
        distances[distances == 0] = 1.0  # elakkan pembahagian sifar
        return np.sum(e * e / m * r_diffs / distances[:, :, np.newaxis] ** 3, axis=1)

    np.random.seed(100)  # benih perawak
    pos = 2 * np.random.random((100, 2)) - 1  # 100 kedudukan 2D secara rawak
    vel = np.zeros((100, 2))  # 100 halaju 2D sifar sebagai permulaan
    # vel = np.apply_along_axis(lambda p: np.array([[0, -1], [1, 0]]) @ p, 1, pos)
    # ^^^ percubaan memperoleh satu pusaran
    xs, vs = [pos], [vel + get_acc(pos) * dt]

    for _ in ts[1:]:
        xs.append(xs[-1] + 2 * vs[-1] * dt)
        vs.append(vs[-1] + 2 * get_acc(xs[-1]) * dt)

    fig, ax = plt.subplots()
    ax.set_title("100 zarah")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    (particles,) = ax.plot(pos[:, 0], pos[:, 1], marker="o", ls="")
    fig.subplots_adjust(bottom=0.25)

    ax_time = fig.add_axes([0.1, 0.1, 0.8, 0.02])
    time_slider = Slider(ax=ax_time, label="t", valmin=0, valmax=10, valinit=0)

    def update(val):
        ind = int(val / 10 * (len(xs) - 1))
        particles.set_data(xs[ind][:, 0], xs[ind][:, 1])

    time_slider.on_changed(update)

    plt.show()


def soalan3(Q, B, W0, n=""):
    ths = [0]
    ws = [0]

    # parameter simulasi
    dt = 0.1
    ts = np.arange(0, 120, dt)  # simulasi selama 2 minit

    def RK(x1, y1, f):
        K1 = f(x1, y1)
        K2 = f(x1 + dt / 2, y1 + K1 * dt / 2)
        K3 = f(x1 + dt / 2, y1 + K2 * dt / 2)
        K4 = f(x1 + dt, y1 + K3 * dt)
        return y1 + (K1 + 2 * K2 + 2 * K3 + K4) * dt / 6

    for t in ts[1:]:
        ths.append(
            RK(
                t,
                ths[-1],
                lambda t, th: ws[-1],
            )
        )
        ws.append(
            RK(
                t,
                ws[-1],
                lambda t, w: -Q * w - np.sin(ths[-1]) + B * np.cos(W0 * t),
            )
        )

    plt.plot(ths[0], ws[0], label="start", color="r", marker="o")
    plt.plot(ths, ws)
    plt.plot(ths[-1], ws[-1], label="end", color="g", marker="o")
    plt.title(f"Ruang Fasa bagi {Q,B,W0}")
    plt.xlabel("theta")
    plt.ylabel("omega")
    plt.legend()
    plt.savefig(f"A2_3_{n}.png")
    plt.show()

    "SET 1 bertelatah berkala, manakala SET 2 bertelatah kalut"


if __name__ == "__main__":
    soalan1()
    soalan2()
    soalan3(0.5, 0.9, 2 / 3, "1")
    soalan3(0.5, 1.15, 2 / 3, "2")
