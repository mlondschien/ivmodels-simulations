import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ivmodels.quadric import Quadric
from ivmodels.tests import anderson_rubin_test
from ivmodels.utils import proj

from ivmodels_simulations.constants import FIGURES_PATH

fig, axes = plt.subplots(ncols=2, figsize=(7, 2.5))
# plt.tight_layout(rect=[0.05, 0.02, 0.88, 0.98])


for idx, cov, title in [
    [0, np.diag([1, 1]), "Technical condition 2 does not hold"],
    [1, np.array([[1, 0.05], [0.05, 1]]), "Technical condition 2 holds"],
]:
    S = np.array([[1.0, 0], [0, 0.5], [0, 0], [1, 0], [0, 1], [0, 0]]) @ cov
    Z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    y = np.array([0, 0, 0, 0, 0, 1.0]).reshape(-1, 1)

    S_proj = proj(Z, S)
    S_orth = S - S_proj

    Sy = np.hstack([S, y])
    Sy_proj = proj(Z, Sy)
    Sy_orth = Sy - Sy_proj

    kappa = scipy.linalg.eigvalsh(
        a=Sy_proj.T @ Sy_proj, b=Sy_orth.T @ Sy_orth, subset_by_index=[0, 0]
    )[0]

    y_proj = proj(Z, y)
    y_orth = y - y_proj

    n, k = Z.shape
    lambdamin = scipy.linalg.eigvalsh(
        a=S_proj.T @ S_proj, b=S_orth.T @ S_orth, subset_by_index=[0, 0]
    )[0]
    alpha = 1 - scipy.stats.chi2(2).cdf((n - k) * lambdamin)
    print(alpha)
    print(1 - alpha)

    alpha_below = 0.69
    alpha_above = 0.68

    quantile_below = scipy.stats.chi2(2).ppf(1 - alpha_below) / (n - k)
    quantile_above = scipy.stats.chi2(2).ppf(1 - alpha_above) / (n - k)

    inverse_ar = Quadric(
        A=S.T @ (S - (1 + lambdamin) * S_orth) + 1e-8 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y_orth.T
        @ y_orth
        * (
            lambdamin - (y_proj.T @ y_proj) / (y_orth.T @ y_orth)
        ),  # -y.T @ y * lambdamin,
    )
    inverse_ar_below = Quadric(
        A=S.T @ (S - (1 + quantile_below) * S_orth) + 1e-8 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y_orth.T
        @ y_orth
        * (quantile_below - (y_proj.T @ y_proj) / (y_orth.T @ y_orth)),
    )
    inverse_ar_above = Quadric(
        A=S.T @ (S - (1 + quantile_above) * S_orth) + 1e-8 * np.diag((1, 1)),
        b=np.zeros(2),
        c=-y_orth.T
        @ y_orth
        * (quantile_above - (y_proj.T @ y_proj) / (y_orth.T @ y_orth)),
    )

    xrange = (-1.5, 1.5)
    yrange = (-15, 15)
    xspace = np.linspace(*xrange, 100)
    yspace = np.linspace(*yrange, 100)

    xx, yy = np.meshgrid(xspace, yspace)
    zz = np.zeros(xx.shape)

    for i in range(len(yspace)):
        for j in range(len(xspace)):
            beta = np.array([xspace[j], yspace[i]])
            zz[i, j] = anderson_rubin_test(
                Z=Z, X=S, y=y, beta=beta, fit_intercept=False
            )[0]

    # norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0.0, vmax=1.0)
    # color_map1 = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     "cut_my_cmap1", my_cmap(np.linspace(0, 0.08, my_cmap.N))
    # )
    # cax1 = plt.axes((0.92, 0.35, 0.025, 0.3))
    # cbar1 = matplotlib.colorbar.ColorbarBase(
    #     cax1,
    #     cmap=color_map1,
    #     norm=norm1,
    # )
    # cbar1.set_ticks([0.0, 0.025, 0.05, 0.075])
    # cbar1.set_ticklabels([0.0, 0.025, 0.05, 0.075])

    im = axes[idx].contourf(xx, yy, 3 * zz, levels=100)  # , norm=norm, cmap="viridis")
    axes[idx].plot(
        inverse_ar_above._boundary()[:, 0],
        inverse_ar_above._boundary()[:, 1],
        color="black",
        linestyle="dotted",
        label=(
            "$\\{\\beta \\mid k \\cdot \\mathrm{AR}(\\beta) < F^{-1}_{\\chi^2(2)}(0.32) \\}$"
            if idx == 0
            else None
        ),
    )
    axes[idx].plot(
        inverse_ar._boundary()[:, 0],
        inverse_ar._boundary()[:, 1],
        color="black",
        label=(
            "$\\{\\beta \\mid k \\cdot \\mathrm{AR}(\\beta) < F^{-1}_{\\chi^2(2)}(0.313) \\}$"
            if idx == 0
            else None
        ),
    )
    axes[idx].plot(
        inverse_ar_below._boundary()[:, 0],
        inverse_ar_below._boundary()[:, 1],
        color="black",
        linestyle="--",
        label=(
            "$\\{\\beta \\mid k \\cdot \\mathrm{AR}(\\beta) < F^{-1}_{\\chi^2(2)}(0.31) \\}$"
            if idx == 0
            else None
        ),
    )

    if idx == 0:
        axes[idx].set_ylabel("$\\beta_2$", rotation=0)

    axes[idx].set_xlabel("$\\beta_1$")
    axes[idx].set_title(title, y=1.01, fontsize=11)
    axes[idx].set_xlim(xrange)
    axes[idx].set_ylim(yrange)

fig.legend(loc="outside lower center", bbox_to_anchor=(0.53, -0.25), ncol=3)

cax = plt.axes((0.93, 0.07, 0.03, 0.815))
cbar = matplotlib.colorbar.Colorbar(cax, cmap=im.cmap, norm=im.norm)
cbar.set_label("$k \\cdot \\mathrm{AR}(\\beta)$", labelpad=-20, y=1.12, rotation=0)

plt.show()
fig.savefig(FIGURES_PATH / "figure_tc2_counterexample.pdf", bbox_inches="tight")
