import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ivmodels import KClass
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import anderson_rubin_test, inverse_anderson_rubin_test, rank_test

from ivmodels_simulations.constants import FIGURES_PATH

k = 3
mw = 1
mx = 1

fig, axes = plt.subplots(ncols=3, figsize=(10, 3.5), gridspec_kw={"wspace": 0.1})
# fig.tight_layout(rect=[0.2, 0.2, 0.6, 0.6])

fig.suptitle("(Subvector) inverse Anderson-Rubin test confidence sets", y=1.01)

for idx, n, alpha, title in [
    [0, 100, 0.2, "$1 - \\alpha = 0.8$"],
    [1, 100, 0.1, "$1 - \\alpha = 0.9$"],
    [2, 100, 0.05, "$1 - \\alpha = 0.95$"],
]:
    Z, X, y, _, W, beta0, gamma0 = simulate_gaussian_iv(
        n=n,
        k=k,
        mx=mx,
        mw=mw,
        include_intercept=False,
        return_beta=True,
        return_gamma=True,
        seed=0,
    )
    S = np.hstack([X, W])

    kappa_subv = 1 + scipy.stats.chi2(k - mw).ppf(1 - alpha) / (n - k)
    kappa_full = 1 + scipy.stats.chi2(k).ppf(1 - alpha) / (n - k)
    kclass_subv = KClass(kappa=kappa_subv, fit_intercept=False).fit(X=S, y=y, Z=Z)
    kclass_full = KClass(kappa=kappa_full, fit_intercept=False).fit(X=S, y=y, Z=Z)
    print(
        f"kappa_subv = {kappa_subv}, kappa_full = {kappa_full}, kappa_liml={KClass(kappa='liml', fit_intercept=False).fit(X=S, y=y, Z=Z).kappa_}"
    )

    inverse_ar = inverse_anderson_rubin_test(Z, S, y, fit_intercept=False, alpha=alpha)
    inverse_ar_1 = inverse_anderson_rubin_test(
        Z, X=X, y=y, W=W, fit_intercept=False, alpha=alpha
    )
    inverse_ar_2 = inverse_anderson_rubin_test(
        Z, X=W, y=y, W=X, fit_intercept=False, alpha=alpha
    )

    print(inverse_ar_1)
    print(inverse_ar_2)

    xrange = (-2, 6)
    yrange = (-13, 13)
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

    # https://stackoverflow.com/a/10642653/10586763
    for _ in range(3):
        im = axes[idx].contourf(xx, yy, zz, levels=100, antialiased=True)

    axes[idx].plot(
        beta0, gamma0, "x", color="red", label="True parameter" if idx == 0 else None
    )
    axes[idx].plot(
        kclass_subv.coef_[0],
        kclass_subv.coef_[1],
        "o",
        color="blue",
        label="K-class estimator (subv)" if idx == 0 else None,
    )
    axes[idx].plot(
        kclass_full.coef_[0],
        kclass_full.coef_[1],
        "o",
        color="green",
        label="K-class estimator (full)" if idx == 0 else None,
    )
    boundary = inverse_ar._boundary()
    axes[idx].plot(
        boundary[:200, 0],
        boundary[:200, 1],
        color="black",
        linestyle="--",
        label="Confidence set for $(\\beta_1, \\beta_2)$" if idx == 0 else None,
    )
    if boundary.shape[0] == 400:
        axes[idx].plot(
            boundary[200:, 0], boundary[200:, 1], color="black", linestyle="--"
        )

    boundary = inverse_ar_1._boundary()
    axes[idx].vlines(
        boundary,
        *yrange,
        color="black",
        linestyle="dotted",
        label="Confidence sets for individual parameters" if idx == 0 else None,
    )

    arrowstyle_closed = "]-[, angleA=180, angleB=180, widthA=0.5, widthB=0.5"
    arrowstyle_open = "-[, angleB=180, widthB=0.5"
    arrowstyle_rightopen = "]-, angleA=180, widthA=0.5"

    # label = "Confidence sets for\nindividual parameters" if idx==0 else None
    eps = (xrange[1] - xrange[0]) / 100
    if inverse_ar_1.volume() < np.inf:
        patch = matplotlib.patches.FancyArrowPatch(
            posA=(boundary[0, 0] - eps, yrange[0]),
            posB=(boundary[1, 0] + eps, yrange[0]),
            arrowstyle=arrowstyle_closed,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )  # , label=label)
        axes[idx].add_patch(patch)
    else:
        patch_1 = matplotlib.patches.FancyArrowPatch(
            posA=(xrange[0] - eps, yrange[0]),
            posB=(boundary[0, 0] + eps, yrange[0]),
            arrowstyle=arrowstyle_open,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )
        patch_2 = matplotlib.patches.FancyArrowPatch(
            posA=(xrange[1], yrange[0]),
            posB=(boundary[1, 0] - eps, yrange[0]),
            arrowstyle=arrowstyle_open,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )
        axes[idx].add_patch(patch_1)
        axes[idx].add_patch(patch_2)

    # if inverse_ar_1.volume() < np.inf:
    #     out = axes[idx].fill_betweenx(yrange, *boundary, facecolor="none", hatch="/", edgecolor="black", alpha=0.25, linestyle="--",  label="Confidence sets for\nindividual parameters")
    # else:
    #     axes[idx].fill_betweenx(yrange, xrange[0], boundary[0], facecolor="none", hatch="/", edgecolor="black", alpha=0.25, linestyle="--",  label="Confidence sets for\nindividual parameters")
    #     axes[idx].fill_betweenx(yrange, xrange[1], boundary[1], facecolor="none", hatch="/", edgecolor="black", alpha=0.25, linestyle="--")

    boundary = inverse_ar_2._boundary()
    axes[idx].hlines(boundary, *yrange, color="black", linestyle="dotted")

    eps = (yrange[1] - yrange[0]) / 100
    if inverse_ar_2.volume() < np.inf:
        patch = matplotlib.patches.FancyArrowPatch(
            posA=(xrange[0], boundary[0, 0] - eps),
            posB=(xrange[0], boundary[1, 0] + eps),
            arrowstyle=arrowstyle_closed,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )
        axes[idx].add_patch(patch)
    else:
        patch_1 = matplotlib.patches.FancyArrowPatch(
            posA=(xrange[0], yrange[0] - eps),
            posB=(xrange[0], boundary[0, 0] + eps),
            arrowstyle=arrowstyle_open,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )
        patch_2 = matplotlib.patches.FancyArrowPatch(
            posA=(xrange[0], yrange[1]),
            posB=(xrange[0], boundary[1, 0] - eps),
            arrowstyle=arrowstyle_open,
            mutation_scale=10,
            color="black",
            clip_on=False,
            lw=3.0,
        )
        axes[idx].add_patch(patch_1)
        axes[idx].add_patch(patch_2)
    # if inverse_ar_2.volume() < np.inf:
    #     axes[idx].fill_between(yrange, *boundary, edgecolor="black", facecolor="none", hatch='\\', alpha=0.25, linestyle="--")
    # else:
    #     axes[idx].fill_between(xrange, yrange[0], boundary[0], edgecolor="black", facecolor="none", hatch='\\', alpha=0.25, linestyle="--")
    #     axes[idx].fill_between(xrange, yrange[1], boundary[1], edgecolor="black", facecolor="none", hatch='\\', alpha=0.25, linestyle="--")

    axes[idx].set_xlim(xrange)
    axes[idx].set_ylim(yrange)

    axes[idx].set_xlabel("$\\beta_1$")
    axes[idx].set_title(title)

    box = axes[idx].get_position()
    axes[idx].set_position([box.x0, 0.12 + box.y0, box.width, box.height * 0.83])

axes[1].set_yticklabels([])
axes[2].set_yticklabels([])

# print(inverse_ar_1._boundary())
# print(inverse_ar_2._boundary())
axes[0].set_ylabel("$\\beta_2$", rotation=0)


fig.legend(
    loc="outside lower center", ncols=3
)  # , bbox_to_anchor=(0., 0, 1., .102), ncols=3, borderaxespad=0.)

cax = plt.axes((0.93, 0.17, 0.03, 0.715))
cbar = matplotlib.colorbar.Colorbar(
    cax, cmap=im.cmap, norm=im.norm
)  # , values=im.cvalues)
cbar.set_label("$\\log(\\mathrm{AR}(\\beta))$", labelpad=-25, y=1.08, rotation=0)
# plt.show()
print(rank_test(Z, S, fit_intercept=False))
fig.savefig(FIGURES_PATH / "figure_inverse_ar_different_alpha.pdf", bbox_inches="tight")
