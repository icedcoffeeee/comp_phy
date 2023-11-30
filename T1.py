print("\nGradient Descent for x^2 - 4x\n")


def derivative(f, x0, dx):
    return (f(x0 + dx / 2) - f(x0 - dx / 2)) / dx


f = lambda x: x**2 - 4 * x
x0 = float(input("insert x0: "))
dx = float(input("insert dx: "))
deriv = derivative(f, x0, dx)

print("x0: ", x0, "\tf'(x0): ", deriv)
while abs(deriv) >= dx:
    x0 += dx * (-1 if deriv > 0 else 1)
    deriv = derivative(f, x0, dx)
    print("x0: ", x0, "\tf'(x0): ", deriv)

print("\n\nApproximation of PI\n")

n = int(input("number of samples: "))
segment_length = abs(1 - complex(-1) ** (1 / n))

print("one segment length: ", segment_length)
print("approximation: ", n * segment_length)

# General Relationship
ns = range(1, 101)
pis = [n * abs(1 - complex(-1) ** (1 / n)) for n in ns]

import matplotlib.pyplot as plt

plt.plot(ns, pis)
plt.show()
