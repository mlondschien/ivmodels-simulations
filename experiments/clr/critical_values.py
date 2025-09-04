import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from ivmodels.tests.conditional_likelihood_ratio import _newton_minimal_root
from numba import njit, prange

fig, ax = plt.subplots(
    nrows=2, ncols=3, figsize=(8.5, 4.5), gridspec_kw={"wspace": 0.1, "hspace": 0.3}
)


@njit
def clr_cdf(
    mx: int,
    k: int,
    lambdas: np.ndarray,
    atol=1e-6,
    num_iter=100,
    num_samples=200_000,
):
    statistics = np.empty(num_samples)

    for i in prange(num_samples):
        np.random.seed(i)
        qx = np.random.standard_normal(mx) ** 2
        q0 = np.sum(np.random.standard_normal(k - mx) ** 2)
        q_sum = np.sum(qx) + q0

        mu_min = _newton_minimal_root(
            q_sum, qx, lambdas + q_sum, atol=atol, num_iter=num_iter
        )
        statistics[i] = q_sum - mu_min

    return statistics


m = 2

z = np.linspace(0, 50, 100)


for idx, (m, k, ax) in enumerate(
    zip([2, 2, 2, 4, 4, 4], [3, 5, 10, 6, 10, 20], ax.flatten())
):
    roots_1 = clr_cdf(mx=m, k=k, lambdas=np.array([5] + [5] * (m - 1)))
    roots_2 = clr_cdf(mx=m, k=k, lambdas=np.array([5] + [50] * (m - 1)))

    roots_4 = clr_cdf(mx=m, k=k, lambdas=np.array([10] + [10] * (m - 1)))
    roots_5 = clr_cdf(mx=m, k=k, lambdas=np.array([10] + [100] * (m - 1)))

    y = np.empty(len(z))

    roots_1_cdf = np.empty(len(z))
    for i, zi in enumerate(z):
        roots_1_cdf[i] = np.mean(roots_1 > zi)

    ax.plot(
        z,
        roots_1_cdf,
        label="$\\Delta\\lambda_1 = \\Delta\\lambda_2 = 5$" if idx == 0 else None,
        color="#004488",
        linestyle="solid",
        zorder=3,
    )

    roots_2_cdf = np.empty(len(z))
    for i, zi in enumerate(z):
        roots_2_cdf[i] = np.mean(roots_2 > zi)

    ax.plot(
        z,
        roots_2_cdf,
        label="$\\Delta\\lambda_1 = 5, \\Delta\\lambda_2 = 50$" if idx == 0 else None,
        color="#004488",
        linestyle=(0, (4, 2)),
        zorder=5,
    )

    roots_4_cdf = np.empty(len(z))
    for i, zi in enumerate(z):
        roots_4_cdf[i] = np.mean(roots_4 > zi)

    ax.plot(
        z,
        roots_4_cdf,
        label="$\\Delta\\lambda_1 = \\Delta\\lambda_2 = 10$" if idx == 0 else None,
        color="#BB5566",
        linestyle="solid",
        zorder=3,
    )

    roots_5_cdf = np.empty(len(z))
    for i, zi in enumerate(z):
        roots_5_cdf[i] = np.mean(roots_5 > zi)

    ax.plot(
        z,
        roots_5_cdf,
        label="$\\Delta\\lambda_1 = 10, \\Delta\\lambda_2 = 100$" if idx == 0 else None,
        color="#BB5566",
        linestyle=(3, (4, 2)),
        zorder=5,
    )

    ax.plot(
        z,
        1 - scipy.stats.chi2(m).cdf(z),
        label="$\\chi^2(m)$" if idx == 0 else None,
        color="#DDAA33",
    )

    ax.axhline(y=0.05, color="black", linestyle="dotted", linewidth=1)

    ax.set_xlabel("z")
    ax.set_ylim(0, 0.2)
    ax.set_xlim(scipy.stats.chi2(m).ppf(0.8) - 0.5, np.min(z[roots_1_cdf < 0.01]))
    ax.set_title(f"$m={m}, k={k}$", fontsize=10)
    if idx <= 2:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("z", labelpad=1)

    if idx % 3 != 0:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("$\\mathrm{P}[\\,\\text{LR}(\\beta_0) > z\\,]$", fontsize=11)

fig.legend(
    loc="lower center",
    bbox_to_anchor=(0.48, -0.07),
    ncol=5,
    columnspacing=1,
    handletextpad=0.4,
)
fig.savefig("subvector_clr.pdf", bbox_inches="tight")
