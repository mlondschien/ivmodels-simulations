import matplotlib.pyplot as plt
import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.simulate import simulate_guggenberger12
from ivmodels.tests.lagrange_multiplier import _LM
from ivmodels.utils import proj

from ivmodels_simulations.constants import FIGURES_PATH

n_seeds = 10
ks = [10, 20, 30]
n = 1000

fig, axes = plt.subplots(nrows=n_seeds, ncols=len(ks), figsize=(10, 10))
xlim = (-3, 5)
for k_idx, k in enumerate(ks):
    axes[0, k_idx].set_title(f"$k={k}$")
    for seed in range(n_seeds):
        Z, X, y, _, W, _, beta0 = simulate_guggenberger12(
            n=n,
            k=k,
            return_beta=True,
            seed=seed,
        )

        gamma0 = np.array([1])
        grid = np.linspace(*xlim, 200)
        statistic = np.zeros_like(grid)

        X_proj, y_proj, W_proj = proj(Z, X, y, W)

        XW = np.hstack([X, W])
        XW_proj = np.hstack([X_proj, W_proj])

        lm = _LM(Z=Z, X=X, y=y, W=W, dof=n - k, gamma_0=["zero", "liml"])

        for idx, x in enumerate(grid):
            statistic[idx] = lm.derivative(
                beta=beta0, gamma=np.array([x]), jac=False, hess=False
            )[0]

        axes[seed, k_idx].plot(
            grid, statistic, label="$\\mathrm{LM}(\\beta_0, \\gamma)$"
        )

        liml = (
            KClass(kappa="liml", fit_intercept=False)
            .fit(X=W, y=y - X @ beta0, Z=Z)
            .coef_
        )
        stat_at_liml = lm.derivative(beta=beta0, gamma=liml, jac=False, hess=False)[0]
        stat_at_truth = lm.derivative(beta=beta0, gamma=gamma0, jac=False, hess=False)[
            0
        ]

        stat_at_min, minimizer = lm.lm(beta=beta0, return_minimizer=True)

        if xlim[0] < liml < xlim[1]:
            axes[seed, k_idx].scatter(
                x=liml,
                y=stat_at_liml,
                color="darkorange",
                marker="x",
                label="LIML",
                zorder=10,
                clip_on=False,
            )

        if xlim[0] < gamma0 < xlim[1]:
            axes[seed, k_idx].scatter(
                x=gamma0,
                y=stat_at_truth,
                color="black",
                marker="o",
                label="$\\gamma_0$",
                zorder=9,
                clip_on=False,
                alpha=0.5,
            )

        if xlim[0] < minimizer < xlim[1]:
            axes[seed, k_idx].scatter(
                x=minimizer,
                y=stat_at_min,
                color="red",
                marker="*",
                label="minimizer",
                zorder=11,
                clip_on=False,
            )
        print(f"seed={seed}, k={k}, minimizer={minimizer}, stat_at_min={stat_at_min}")

        axes[seed, k_idx].set_xlim(xlim)
        axes[seed, k_idx].set_ylim(0, np.max(statistic) * 1.05)
        axes[seed, k_idx].axhline(
            y=scipy.stats.chi(1).ppf(0.95),
            color="black",
            linestyle="dotted",
            label="$F_{\\chi^2(1)}^{-1}(0.95)$",
        )

        if seed < n_seeds - 1:
            axes[seed, k_idx].set_xticklabels([])

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.03))
fig.savefig(FIGURES_PATH / "optimization.pdf", bbox_inches="tight")
