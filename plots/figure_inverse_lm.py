import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import lagrange_multiplier_test, inverse_anderson_rubin_test, rank_test

from ivmodels_simulations.constants import FIGURES_PATH

n = 100
k = 3
mw = 1
mx = 1
mc = 3
u = 2

alpha = 0.05

Z, X, y, C, W, beta0, gamma0 = simulate_gaussian_iv(
    n=n,
    k=k,
    mx=mx,
    mw=mw,
    include_intercept=True,
    return_beta=True,
    return_gamma=True,
    seed=0,
)

S = np.hstack([X, W])

xrange = (-10, 10)
yrange = (-12, 5)
xspace = np.linspace(*xrange, 500)
yspace = np.linspace(*yrange, 500)

xx, yy = np.meshgrid(xspace, yspace)
zz = np.zeros(xx.shape)

for i in range(len(yspace)):
    for j in range(len(xspace)):
        beta = np.array([xspace[j], yspace[i]])
        zz[i, j] = np.log(
            lagrange_multiplier_test(Z, S, y, beta, fit_intercept=False)[0]
        )

zz[zz < np.log(scipy.stats.chi(1).ppf(1 - alpha))] = np.nan

fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
im = ax.contourf(xx, yy, zz)

fig.colorbar(im, ax=ax)
plt.show()