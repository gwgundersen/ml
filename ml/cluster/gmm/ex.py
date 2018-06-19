import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def q(x, y):
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6 * g1 + 28.4 * g2 / (0.6 + 28.4)


dx = 0.01
x = np.arange(-4, 4, dx)
y = np.arange(-4, 4, dx)
X, Y = np.meshgrid(x, y)
Z = q(X, Y)

plt.contour(X, Y, Z, 10, alpha=0.5)
plt.savefig("samples.png")
