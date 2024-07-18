from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12
from ivmodels.tests import lagrange_multiplier_test, inverse_wald_test, wald_test, anderson_rubin_test, conditional_likelihood_ratio_test
import matplotlib.pyplot as plt

from functools import partial
from ivmodels.tests.lagrange_multiplier import _LM
from ivmodels.confidence_set import ConfidenceSet
import numpy as np
import scipy
from ivmodels import KClass
import click
from ivmodels_simulations.constants import FIGURES_PATH
from ivmodels_simulations.tests import lagrange_multiplier_test_one_step    


COLORS = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "black": "#000000",
    "indigo": "#332288",
    "teal": "#44AA99",
}


@click.command()
@click.option("--n", default=1000)
@click.option("--seed", default=0)
@click.option("--k", default=10)
@click.option("--data", default="gaussian")
@click.option("--betas", default=None)
@click.option("--f_", default="p-value")
def main(n, seed, k, data, betas, f_):
    if data == "gaussian":
        Z, X, y, C, W, beta0, gamma0 = simulate_gaussian_iv(
            n=n,
            k=k,
            mx=1,
            mw=1,
            include_intercept=False,
            return_beta=True,
            return_gamma=True,
            seed=seed,
        )
        grid = np.linspace(-2, 2, 100) + beta0.item()

    elif data=="guggenberger12":
        Z, X, y, C, W, beta0 = simulate_guggenberger12(
            n=n,
            k=k,
            return_beta=True,
            seed=seed,
        )
        gamma0 = np.array([1])
        grid = np.linspace(-1, 1, 100) + beta0.item()

    tests = {
        **{
            f"{method}, {gamma_0}": partial(lagrange_multiplier_test, optimizer=method, gamma_0=[gamma_0])
            for gamma_0 in ["liml"]
            for method in [
                "newton-cg",
                "bfgs"
            ]
        },
        **{
            "1 - truth - liml": partial(lagrange_multiplier_test_one_step, ddlm="truth", gamma0="liml"),
            "1 - truth - gamma0": partial(lagrange_multiplier_test_one_step, ddlm="truth", gamma0=gamma0.flatten()),
            "1 - ar - liml": partial(lagrange_multiplier_test_one_step, ddlm="ar", gamma0="liml"),
            "1 - ar - gamma0": partial(lagrange_multiplier_test_one_step, ddlm="ar", gamma0=gamma0.flatten()),
            "1 - kappa - liml": partial(lagrange_multiplier_test_one_step, ddlm="kappa", gamma0="liml"),
            "1 - kappa - gamma0": partial(lagrange_multiplier_test_one_step, ddlm="kappa", gamma0=gamma0.flatten()),
        }
    }

    COLOR_MAPPING = {
        test_name: color for test_name, color in zip(tests.keys(), COLORS.values())
    }

    def f(x):
        if f_ == "p-value":
            return 1 - scipy.stats.chi2(1).cdf(x)
        elif f_ == "log(stat)":
            return np.log(1e-6 + x)
        elif f_ == "stat":
            return x

    if betas is None:
        betas = [beta0.item()]
    else:
        betas = [beta0.item()] + list(map(float, betas.split(",")))

    fig, axes = plt.subplots(len(betas) + 1, 1, figsize=(10, 10))

    for ldx, (test_name, test) in enumerate(tests.items()):
        y_ = np.zeros_like(grid)
        for jdx, x in enumerate(grid):
            y_[jdx] = f(test(Z=Z, X=X, y=y, W=W, beta=np.array([x]), C=C, fit_intercept=False)[0])

        axes[0].plot(grid, y_, label=test_name, ls=['-','--','-.',':'][ldx%4], color=COLOR_MAPPING[test_name])
        axes[0].set_ylabel(f_)

    axes[0].axvline(x=beta0[0], color="black", linestyle="--", label="True beta")

    for beta in betas[1:]:
        axes[0].axvline(x=beta, color="grey", linestyle="--", alpha=0.5)

    for idx, beta in enumerate(betas):
        idx += 1

        print(f"{idx}/{len(betas)}, beta={beta}")

        min_ = 0
        max_ = 0
        for ldx, (test_name, test) in enumerate(tests.items()):
            result = test(Z=Z, X=X, y=y, W=W, beta=np.array([beta]), C=C, fit_intercept=False)
            print(f"{test_name}: {result}")
            axes[idx].scatter(result[2], f(result[0]), label=test_name, color=COLOR_MAPPING[test_name], marker=["o", "x", "+", "*"][ldx%4])
            min_ = min(min_, result[2])
            max_ = max(max_, result[2])
        
        min_ = min_ - 0.1 * (max_ - min_)
        max_ = max_ + 0.1 * (max_ - min_)
        
        grid = np.linspace(min_ - 1, max_ + 1, 200)

        y_ = np.zeros_like(grid)
        for jdx, g in enumerate(grid):
            res = lagrange_multiplier_test(Z, X=np.hstack([X, W]), C=C, y=y, beta=np.array([beta, g.item()]), fit_intercept=False)
            y_[jdx] = f(res[0]) # 1 - scipy.stats.chi2(1).cdf(res[0])
        
        liml=KClass(kappa="liml", fit_intercept=False).fit(X=W, y=y - X @ np.array([beta]), Z=Z).coef_
        axes[idx].axvline(x=liml, color="black", linestyle="--", label="liml")
        res = _LM(Z=Z, X=X, y=y, W=W, dof=n-k).derivative(beta=np.array([beta]), gamma=liml)
        # axes[idx].axvline(x=liml - res[1], color="black", linestyle="dotted", label="liml - newton")

        axes[idx].plot(grid, y_, label="lm", color="black")
        axes[idx].set_ylabel(f_)
        if idx == 1:
            axes[idx].set_title("beta = beta0")
        else:
            axes[idx].set_title(f"beta = {beta}")

    axes[-1].legend()
    fig.suptitle(f"n={n}, k={k}, data={data}, seed={seed}")
    # fig.savefig(FIGURES_PATH / "optimization" / f"lm_by_beta_gamma_{n}_{k}_{data}_{seed}.pdf")

    plt.show()

if __name__ == "__main__":
    main()