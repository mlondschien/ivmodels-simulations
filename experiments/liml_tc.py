import numpy as np
from ivmodels.utils import proj
from scipy.optimize import minimize
from ivmodels.tests import anderson_rubin_test
import matplotlib.pyplot as plt

# X = np.array([[1, 1], [1, 1], [0, 1], [0, 0], [0, 0]])
# Z = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 0]])
# y = np.array([0, 0, 1, 1, 1]).reshape(-1, 1)
X = np.array([[0.5, 0], [0, 1], [0, 0], [1, 0], [0, 1], [0, 0]])
y = np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1)
Z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

Xy = np.concatenate([X, y], axis=1)

Xy_proj = proj(Z, Xy)
Xy_orth = Xy - Xy_proj

X_proj = Xy_proj[:, :-1]
X_orth = Xy_orth[:, :-1]

y_proj = Xy_proj[:, -1:]
y_orth = Xy_orth[:, -1:]

print(np.linalg.eig( np.linalg.solve(Xy_orth.T @ Xy_orth, Xy.T @ Xy) ))
kappa_liml = np.min(np.linalg.eig( np.linalg.solve(Xy_orth.T @ Xy_orth, Xy.T @ Xy) )[0])
lambda_1 = np.min( np.linalg.eig( np.linalg.solve(X_orth.T @ X_orth, X.T @ X) )[0] )

X_kappa = kappa_liml * X_proj + (1 - kappa_liml) * X
Xy_kappa = kappa_liml * Xy_proj + (1 - kappa_liml) * Xy

breakpoint()


# beta_liml = np.linalg.solve( X_kappa.T @ X, X_kappa.T @ y )

print(f"Kappa LIML: {kappa_liml}")
print(f"Lambda 1: {lambda_1}")
# print(f"Beta LIML: {beta_liml}")
# print(f"AR(beta_liml) = {anderson_rubin_test(Z, X, y, beta_liml, fit_intercept=False)[0]}")
res = minimize(lambda beta: anderson_rubin_test(Z, X, y, beta, fit_intercept=False)[0], np.ones(X.shape[1]).flatten())
print(res)
x_ = np.linspace(-5, 5, 100)
y_ = np.linspace(-5, 5, 100)

xx, yy = np.meshgrid(x_, y_)
zz = np.zeros(xx.shape)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i, j] = np.log10(max(anderson_rubin_test(Z, X, y=y, W=None, beta=np.array([xx[i, j], yy[i, j]]), fit_intercept=False)[0], 1e-16))

fig, ax = plt.subplots(figsize=(10, 10), ncols=1)

im = ax.contourf(xx, yy, zz)
fig.colorbar(im, ax=ax)
ax.plot(res.x[0], res.x[1], "ro")
plt.show()