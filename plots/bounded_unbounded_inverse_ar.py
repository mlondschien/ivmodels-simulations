import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ivmodels.quadric import Quadric
from ivmodels.tests import anderson_rubin_test
from ivmodels.utils import proj

from ivmodels_simulations.constants import FIGURES_PATH

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
plt.tight_layout(rect=[0.05, 0.02, 0.88, 0.98])


for idx, cov, title in [
    [0, np.diag([1, 1]), "Technical condition 2 does not hold"],
    [1, np.array([[1, 0.05], [0.05, 1]]), "Technical condition 2 holds"],
]:
    S = np.array([[1, 0], [0, 0.5], [0, 0], [1, 0], [0, 1], [0, 0]]) @ cov
    Z = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    y = np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1)

    S_proj = proj(Z, S)
    S_orth = S - S_proj

    n, k = Z.shape
    lambdamin = scipy.linalg.eigvalsh(
        a=S_proj.T @ S_proj, b=S_orth.T @ S_orth, subset_by_index=[0, 0]
    )[0]
    alpha = 1 - scipy.stats.chi2(1).cdf((n - k) * lambdamin)

    alpha_below = 0.31
    alpha_above = 0.33

    quantile_below = scipy.stats.chi2(1).ppf(1 - alpha_below) / (n - k)
    quantile_above = scipy.stats.chi2(1).ppf(1 - alpha_above) / (n - k)

    inverse_ar = Quadric(
        A=S.T @ (S_proj - lambdamin * S_orth) + 1e-6 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y.T @ y * lambdamin,
    )
    inverse_ar_below = Quadric(
        A=S.T @ (S_proj - quantile_below * S_orth) + 1e-6 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y.T @ y * quantile_below,
    )
    inverse_ar_above = Quadric(
        A=S.T @ (S_proj - quantile_above * S_orth) + 1e-6 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y.T @ y * quantile_above,
    )
    print(f"lambdamin: {lambdamin}, alpha: {alpha}")

    xrange = (-3, 3)
    yrange = (-15, 15)
    xspace = np.linspace(*xrange, 100)
    yspace = np.linspace(*yrange, 100)

    xx, yy = np.meshgrid(xspace, yspace)
    zz = np.zeros(xx.shape)

    for i in range(len(yspace)):
        for j in range(len(xspace)):
            beta = np.array([xspace[j], yspace[i]])
            zz[i, j] = np.log(
                anderson_rubin_test(Z, S, y, beta, fit_intercept=False)[0]
            )

    im = axes[idx].contourf(xx, yy, zz, levels=100)

    axes[idx].plot(
        inverse_ar_above._boundary()[:, 0],
        inverse_ar_above._boundary()[:, 1],
        color="black",
        linestyle="dotted",
        label="$\\{\\beta \\mid \\mathrm{AR}(\\beta) < F_{\\chi^2(1)}(0.67) \\}$",
    )
    axes[idx].plot(
        inverse_ar._boundary()[:, 0],
        inverse_ar._boundary()[:, 1],
        color="black",
        label="$\\{\\beta \\mid \\mathrm{AR}(\\beta) < F_{\\chi^2(1)}(0.683) \\}$",
    )
    axes[idx].plot(
        inverse_ar_below._boundary()[:, 0],
        inverse_ar_below._boundary()[:, 1],
        color="black",
        linestyle="--",
        label="$\\{\\beta \\mid \\mathrm{AR}(\\beta) < F_{\\chi^2(1)}(0.69) \\}$",
    )

    if idx == 0:
        axes[idx].set_ylabel("$\\beta_2$", rotation=0)

    axes[idx].set_xlabel("$\\beta_1$")
    axes[idx].set_title(title, y=1.01)
    axes[idx].set_xlim(xrange)
    axes[idx].set_ylim(yrange)

axes[idx].legend(loc="lower right")

cax = plt.axes((0.88, 0.11, 0.03, 0.815))
cbar = matplotlib.colorbar.Colorbar(
    cax, cmap=im.cmap, norm=im.norm
)  # , values=im.cvalues)
cbar.set_label("$\\log(\\mathrm{AR}(\\beta))$", labelpad=-20, y=1.08, rotation=0)
# plt.show()
# fig.savefig(FIGURES_PATH / "figure_tc2_counterexample.eps")
