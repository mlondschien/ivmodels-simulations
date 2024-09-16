import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ivmodels.tests import anderson_rubin_test, inverse_anderson_rubin_test

from ivmodels_simulations.constants import FIGURES_PATH

n = 100
k = 3
mw = 1
mx = 1

fig, axes = plt.subplots(ncols=3, figsize=(10, 3.5), gridspec_kw={"wspace": 0.1})

fig.suptitle("(Subvector) inverse Anderson-Rubin test confidence sets", y=1.01)


# From ivmodels=0.3.0.
def simulate_gaussian_iv(
    n,
    *,
    mx,
    k,
    u=None,
    mw=0,
    mc=0,
    seed=0,
    include_intercept=True,
    return_beta=False,
    return_gamma=False,
):
    """
    Simulate a Gaussian IV dataset.

    Parameters
    ----------
    n : int
        Number of observations.
    mx : int
        Number of endogenous variables.
    k : int
        Number of instruments.
    u : int, optional
        Number of unobserved variables. If None, defaults to mx.
    mw : int, optional
        Number of endogenous variables not of interest.
    mc : int, optional
        Number of exogenous included variables.
    seed : int, optional
        Random seed.
    include_intercept : bool, optional
        Whether to include an intercept.
    return_beta : bool, optional
        Whether to return the true beta.
    return_gamma : bool, optional
        Whether to return the true gamma.

    Returns
    -------
    Z : np.ndarray of dimension (n, k)
        Instruments.
    X : np.ndarray of dimension (n, mx)
        Endogenous variables.
    y : np.ndarray of dimension (n,)
        Outcomes.
    C : np.ndarray of dimension (n, mc)
        Exogenous included variables.
    W : np.ndarray of dimension (n, mw)
        Endogenous variables not of interest.
    beta : np.ndarray of dimension (mx,)
        True beta. Only returned if ``return_beta`` is True.
    gamma : np.ndarray of dimension (mw,)
        True gamma. Only returned if ``return_gamma`` is True.
    """
    rng = np.random.RandomState(seed)
    beta = rng.normal(0, 1, (mx, 1))

    if u is None:
        u = mx

    ux = rng.normal(0, 1, (u, mx))
    uy = rng.normal(0, 1, (u, 1))
    uw = rng.normal(0, 1, (u, mw))

    alpha = rng.normal(0, 1, (mc, 1))
    gamma = rng.normal(0, 1, (mw, 1))

    Pi_ZX = rng.normal(0, 1, (k, mx))
    Pi_ZW = rng.normal(0, 1, (k, mw))
    Pi_CX = rng.normal(0, 1, (mc, mx))
    Pi_CW = rng.normal(0, 1, (mc, mw))
    Pi_CZ = rng.normal(0, 1, (mc, k))

    U = rng.normal(0, 1, (n, u))
    C = rng.normal(0, 1, (n, mc)) + include_intercept * rng.normal(0, 1, (1, mc))

    Z = (
        rng.normal(0, 1, (n, k))
        + include_intercept * rng.normal(0, 1, (1, k))
        + C @ Pi_CZ
    )

    X = Z @ Pi_ZX + C @ Pi_CX + U @ ux
    X += rng.normal(0, 1, (n, mx)) + include_intercept * rng.normal(0, 1, (1, mx))
    W = Z @ Pi_ZW + C @ Pi_CW + U @ uw
    W += rng.normal(0, 1, (n, mw)) + include_intercept * rng.normal(0, 1, (1, mw))
    y = C @ alpha + X @ beta + W @ gamma + U @ uy
    y += rng.normal(0, 1, (n, 1)) + include_intercept * rng.normal(0, 1, (1, 1))

    if return_beta and return_gamma:
        return Z, X, y.flatten(), C, W, beta.flatten(), gamma.flatten()
    elif return_beta:
        return Z, X, y.flatten(), C, W, beta.flatten()
    elif return_gamma:
        return Z, X, y.flatten(), C, W, gamma.flatten()
    else:
        return Z, X, y.flatten(), C, W


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


for idx, n, alpha, title in [
    [0, 100, 0.2, "$1 - \\alpha = 0.8$"],
    [1, 100, 0.1, "$1 - \\alpha = 0.9$"],
    [2, 100, 0.05, "$1 - \\alpha = 0.95$"],
]:
    inverse_ar_1 = inverse_anderson_rubin_test(
        Z, X=X, y=y, W=W, fit_intercept=False, alpha=alpha
    )
    inverse_ar_2 = inverse_anderson_rubin_test(
        Z, X=W, y=y, W=X, fit_intercept=False, alpha=alpha
    )
    inverse_ar = inverse_anderson_rubin_test(Z, S, y, fit_intercept=False, alpha=alpha)

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
        )
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

    axes[idx].set_xlim(xrange)
    axes[idx].set_ylim(yrange)

    axes[idx].set_xlabel("$\\beta_1$")
    axes[idx].set_title(title)

    box = axes[idx].get_position()
    axes[idx].set_position([box.x0, 0.12 + box.y0, box.width, box.height * 0.83])

axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
axes[0].set_ylabel("$\\beta_2$", rotation=0)


fig.legend(loc="outside lower center", ncols=3)

cax = plt.axes((0.93, 0.17, 0.03, 0.715))
cbar = matplotlib.colorbar.Colorbar(cax, cmap=im.cmap, norm=im.norm)
cbar.set_label("$\\log(\\mathrm{AR}(\\beta))$", labelpad=-25, y=1.08, rotation=0)

plt.show()
fig.savefig(FIGURES_PATH / "figure_inverse_ar_different_alpha.pdf", bbox_inches="tight")
