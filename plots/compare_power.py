from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12
from ivmodels.tests import lagrange_multiplier_test, inverse_wald_test, wald_test, anderson_rubin_test, conditional_likelihood_ratio_test
from functools import partial
import matplotlib.pyplot as plt
from ivmodels.confidence_set import ConfidenceSet
import numpy as np
import click
from ivmodels_simulations.constants import FIGURES_PATH

tests = {
    f"lm ({method}, {gamma_0})": partial(lagrange_multiplier_test, optimizer=method, gamma_0=[gamma_0])
    for gamma_0 in ["zero", "liml"]
    for method in [
        "cg",
        "newton-cg",
        "trust-exact",
        "bfgs"
        ]
}
tests["clr"] = conditional_likelihood_ratio_test
tests["ar"] = anderson_rubin_test
tests["wald (liml)"] = partial(wald_test, estimator="liml")

@click.command()
@click.option("--mw", default=1)
@click.option("--n_plots", default=5)
@click.option("--data", default="gaussian")
@click.option("--n", default=100)
@click.option("--k", default=10)
def main(mw, n_plots, data, n, k):

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 10))

    results = {idx: {} for idx in range(n_plots)}

    for idx in range(n_plots):
        if data == "gaussian":
            Z, X, y, C, W, beta0 = simulate_gaussian_iv(
                n=n,
                k=k,
                mx=1,
                mw=mw,
                include_intercept=True,
                return_beta=True,
                seed=idx,
            )
            grid = np.linspace(-2, 2, 200) + beta0[0]
        elif data=="guggenberger12":
            if mw != 1:
                raise ValueError("mw must be 1 for Guggenberger12 data.")

            Z, X, y, C, W, beta0 = simulate_guggenberger12(
                n=n,
                k=k,
                return_beta=True,
                seed=idx,
            )
            grid = np.linspace(-1, 1, 200) + beta0[0]

        for ldx, (test_name, test) in enumerate(tests.items()):
            y_ = np.zeros_like(grid)
            for jdx, x in enumerate(grid):
                y_[jdx] = test(Z=Z, X=X, y=y, W=W, beta=np.array([x]), C=C, fit_intercept=True)[1]
            results[idx][test_name] = y_

            axes[idx].plot(grid, y_, label=test_name, ls=['-','--','-.',':'][ldx%4])
            axes[idx].set_ylabel("p-value")
        axes[idx].axvline(x=beta0[0], color="black", linestyle="--", label="True beta")

    axes[-1].legend()

    fig.suptitle(f"data={data}, n={n}, k={k}, mw={mw}")

    fig.savefig(FIGURES_PATH / "optimization" / f"power_comparison_{n}_{k}_{mw}_{data}.pdf")

if __name__ == "__main__":
    main()