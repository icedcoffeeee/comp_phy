import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Q1:
    def __init__(self):
        V, E = self.init()

        ortho_v, ortho_e = self.ortho_vertices(V, E)
        plt.plot(*ortho_e.T, color="k")
        plt.scatter(*ortho_v.T, color="b")
        plt.show()

        persp_v, persp_e = self.persp_vertices(V, E)
        plt.plot(*persp_e.T, color="k")
        plt.scatter(*persp_v.T, color="b")
        plt.show()

        print("i:\n", V + np.array([1.1, 0.0, 2.5]), "\n")

        # ii
        R = [0, 1, 0]
        print("ii:\n", self.rotate(V, np.pi / 8, R), "\n")

        # iii
        P = [-1.1, 0.0, -2.5]
        iii = self.rotate(V - P, np.pi / 8, R) + P
        print("iii:\n", iii, "\n")

        # iv
        R = [1.1, 1.1, 0]
        P = [0, 1.1, -2.5]
        iv = self.rotate(V - P, np.pi / 8, R) + P
        print("iv:\n", iv, "\n")

        ortho_v, ortho_e = self.ortho_vertices(iii, E)
        plt.plot(*ortho_e.T, color="k")
        plt.scatter(*ortho_v.T, color="b")
        plt.show()

        ortho_v, ortho_e = self.ortho_vertices(iv, E)
        plt.plot(*ortho_e.T, color="k")
        plt.scatter(*ortho_v.T, color="b")
        plt.show()

    def init(self):
        # fmt: off
        return np.float64([
            [3.1, 3.1, 1.3],
            [0.3, 0.3, 0.2],
            [2.3, 5.2, 0.4],
            [5.4, 4.0, 0.5],
            [5.6, 1.1, 0.3],
            [2.6, 0.2, 0.2],
        ]), np.int32([
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
            [2, 3], [3, 4], [4, 5], [5, 6], [6, 2],
        ])
        # fmt: on

    def init_cube(self):
        # fmt: off
        return np.float64([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]), np.int32([
            [1, 2], [1, 3], [1, 5],
            [2, 6], [2, 4],
            [3, 4], [3, 7],
            [5, 6], [5, 7],
            [4, 8], [6, 8], [7, 8],
        ])
        # fmt: on

    def ortho_vertices(self, verts: np.ndarray, E: np.ndarray):
        ortho_v = verts[:, :2]
        ortho_e = np.float64([[ortho_v[i], ortho_v[j]] for i, j in E - 1])
        return ortho_v, ortho_e

    def persp_vertices(self, verts: np.ndarray, E: np.ndarray, Z: float = -5):
        pers_v = -Z * np.apply_along_axis(lambda v: v / (v[-1] - Z), 1, verts)[:, :2]
        pers_e = np.float64([[pers_v[i], pers_v[j]] for i, j in E - 1])
        return pers_v, pers_e

    def rotate(self, verts: np.ndarray, angle: float, axis: np.ndarray):
        axis /= np.linalg.norm(axis)
        return np.apply_along_axis(
            lambda v: np.cos(angle) * v
            + np.sin(angle) * np.cross(axis, v)
            + (1 - np.cos(angle)) * np.outer(axis, axis) @ v,
            1,
            verts,
        )


class Q2:
    def __init__(self):
        im = Image.open("image001.png")
        im.show("before")
        arr = np.asarray(im).astype("f8") / 255
        arr = np.pad(arr, 1, "edge")  # padding for edge cases

        arr2 = np.zeros(im.size)
        for i in range(im.height):
            for j in range(im.width):
                mat = arr[i : i + 3, j : j + 3]  # 3x3 matrix
                arr2[i, j] = np.median(mat)

        im2 = Image.fromarray(np.uint8(arr2 * 255))
        im2.save("image002.png")
        im2.show("after")

        arr2 = np.pad(arr2, [[1, 0], [1, 0]])
        Gx = np.zeros(im.size)
        Gy = np.zeros(im.size)
        for i in range(im.height):
            for j in range(im.width):
                Gx[i, j] = arr2[i, j + 1] - arr[i, j]
                Gy[i, j] = arr2[i + 1, j] - arr[i, j]

        thresh_arr = abs(Gx) + abs(Gy)
        thresh = np.median(thresh_arr)
        thresh_arr[thresh_arr < thresh] = 0
        thresh_arr[thresh_arr >= thresh] = 1
        im3 = Image.fromarray(np.uint8(thresh_arr * 255))
        im3.save("image003.png")
        im3.show()


if __name__ == "__main__":
    Q1()
    Q2()
